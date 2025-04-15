@testset "optimize m50" begin
    step_func(x) = (tanh(5.0 * x) + 1.0) * 0.5

    # Model parameters
    # Physical process parameters
    @parameters Tmin Tmax Df Smax f Qmax
    # Normalization parameters
    @parameters snowpack_std snowpack_mean
    @parameters soilwater_std soilwater_mean
    @parameters prcp_std prcp_mean
    @parameters temp_std temp_mean

    # Model variables
    # Input variables
    @variables prcp temp lday
    # State variables
    @variables snowpack soilwater
    # Process variables
    @variables pet rainfall snowfall melt
    # Neural network variables
    @variables log_evap_div_lday log_flow flow
    @variables norm_snw norm_slw norm_temp norm_prcp

    # Neural network definitions
    ep_nn = Lux.Chain(
        Lux.Dense(3 => 16, tanh),
        Lux.Dense(16 => 1, leakyrelu),
        name=:epnn
    )
    ep_nn_params = ComponentVector(LuxCore.initialparameters(StableRNG(42), ep_nn)) |> Vector

    q_nn = Lux.Chain(
        Lux.Dense(2 => 16, tanh),
        Lux.Dense(16 => 1, leakyrelu),
        name=:qnn
    )
    q_nn_params = ComponentVector(LuxCore.initialparameters(StableRNG(42), q_nn)) |> Vector

    # Snow component
    snow_bucket = @hydrobucket :m50_snow begin
        fluxes = begin
            @hydroflux pet ~ 29.8 * lday * 24 * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
            @hydroflux begin
                snowfall ~ step_func(Tmin - temp) * prcp
                rainfall ~ step_func(temp - Tmin) * prcp
            end
            @hydroflux melt ~ step_func(temp - Tmax) * min(snowpack, Df * (temp - Tmax))
        end
        dfluxes = begin
            @stateflux snowpack ~ snowfall - melt
        end
    end

    ep_nn_flux = @neuralflux log_evap_div_lday ~ ep_nn([norm_snw, norm_slw, norm_temp])

    # Soil water component
    soil_bucket = @hydrobucket :m50_soil begin
        fluxes = begin
            @hydroflux norm_snw ~ (snowpack - snowpack_mean) / snowpack_std
            @hydroflux norm_slw ~ (soilwater - soilwater_mean) / soilwater_std
            @hydroflux norm_prcp ~ (prcp - prcp_mean) / prcp_std
            @hydroflux norm_temp ~ (temp - temp_mean) / temp_std
            @neuralflux log_evap_div_lday ~ ep_nn([norm_snw, norm_slw, norm_temp])
            @neuralflux log_flow ~ q_nn([norm_slw, norm_prcp])
        end
        dfluxes = begin
            @stateflux soilwater ~ rainfall + melt - step_func(soilwater) * lday * exp(log_evap_div_lday) - step_func(soilwater) * exp(log_flow)
        end
    end

    # Flow conversion
    flow_conversion = @hydroflux flow ~ exp(log_flow)

    # Complete model
    m50_model = @hydromodel :m50 begin
        snow_bucket
        soil_bucket
        flow_conversion
    end

    # predefine the parameters
    f, Smax, Qmax, Df, Tmax, Tmin = 0.01674478, 1709.461015, 18.46996175, 2.674548848, 0.175739196, -2.092959084
    # load data
    file_path = "../data/m50/01013500.csv"
    data = CSV.File(file_path)
    df = DataFrame(data)
    ts = collect(1:100)
    # cols: Baseflow,Evap,Flow,Infiltration,Lday,Melt,Pet,Prcp,Rainfall,Snowfall,Surfaceflow,Temp,SoilWater,SnowWater
    lday_vec = df[ts, "Lday"]
    prcp_vec = df[ts, "Prcp"]
    temp_vec = df[ts, "Temp"]
    flow_vec = df[ts, "Flow"]

    log_flow_vec = log.(flow_vec)
    log_evap_div_lday_vec = log.(df[ts, "Evap"] ./ lday_vec)
    norm_prcp_vec = (prcp_vec .- mean(prcp_vec)) ./ std(prcp_vec)
    norm_temp_vec = (temp_vec .- mean(temp_vec)) ./ std(temp_vec)
    norm_snw_vec = (df[ts, "SnowWater"] .- mean(df[ts, "SnowWater"])) ./ std(df[ts, "SnowWater"])
    norm_slw_vec = (df[ts, "SoilWater"] .- mean(df[ts, "SoilWater"])) ./ std(df[ts, "SoilWater"])
    nn_input = (norm_snw=norm_snw_vec, norm_slw=norm_slw_vec, norm_temp=norm_temp_vec, norm_prcp=norm_prcp_vec)

    @testset "ep_nn train" begin
        ntp_pre = HydroModelTools.NamedTuplePreprocessor(ep_nn_flux.infos)
        ntp_post = HydroModelTools.NamedTuplePostprocessor(ep_nn_flux.infos)
        opt_func(i, p; kw...) = begin
            @pipe (i, p) |>
                  ntp_pre(_[1], _[2]) |>
                  ep_nn_flux(_[1], _[2]; kw...) |>
                  ntp_post(_)
        end
        ep_grad_opt = HydroOptimizer(component=opt_func, solve_alg=Adam(1e-2), adtype=Optimization.AutoZygote(), maxiters=100)
        ep_input_matrix = Matrix(reduce(hcat, collect(nn_input[HydroModels.get_input_names(ep_nn_flux)]))')
        ep_output = (log_evap_div_lday=log_evap_div_lday_vec,)

        ep_opt_params, epnn_loss_df = ep_grad_opt(
            [nn_input[HydroModels.get_input_names(ep_nn_flux)]], [ep_output],
            tunable_pas=ComponentVector(nns=(epnn=ep_nn_params,)),
            const_pas=ComponentVector(),
            return_loss_df=true
        )
        @test true
    end
    @testset "m50 train" begin
        ntp_pre = HydroModelTools.NamedTuplePreprocessor(m50_model.infos)
        ntp_post = HydroModelTools.NamedTuplePostprocessor(m50_model.infos)
        opt_func(i, p; kw...) = begin
            @pipe (i, p) |>
                  ntp_pre(_[1], _[2]) |>
                  m50_model(_[1], _[2]; kw...) |>
                  ntp_post(_)
        end
        m50_opt = HydroModelTools.HydroOptimizer(component=opt_func, solve_alg=Adam(1e-2), adtype=Optimization.AutoZygote(), maxiters=10)
        config = (solver=HydroModelTools.ODESolver(sensealg=BacksolveAdjoint(autodiff=EnzymeVJP())), interp=LinearInterpolation)
        norm_pas = ComponentVector(
            snowpack_mean=mean(norm_snw_vec), soilwater_mean=mean(norm_slw_vec), prcp_mean=mean(norm_prcp_vec), temp_mean=mean(norm_temp_vec),
            snowpack_std=std(norm_snw_vec), soilwater_std=std(norm_slw_vec), prcp_std=std(norm_prcp_vec), temp_std=std(norm_temp_vec)
        )
        m50_const_pas = ComponentVector(
            params=ComponentVector(Tmin=Tmin, Tmax=Tmax, Df=Df; norm_pas...)
        )
        m50_tunable_pas = ComponentVector(
            nns=ComponentVector(epnn=ep_nn_params, qnn=q_nn_params)
        )
        m50_input = (prcp=prcp_vec, lday=lday_vec, temp=temp_vec)
        q_output = (log_flow=flow_vec,)
        run_kwargs = (config=config, initstates=ComponentVector(snowpack=0.0, soilwater=1300.0))
        m50_opt_params, m50_loss_df = m50_opt(
            [m50_input], [q_output],
            tunable_pas=m50_tunable_pas,
            const_pas=m50_const_pas,
            run_kwargs=[run_kwargs],
            return_loss_df=true
        )
        @test true
    end
end