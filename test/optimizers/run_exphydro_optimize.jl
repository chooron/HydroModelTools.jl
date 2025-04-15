@testset "optimize exphydro" begin
    step_func(x) = (tanh(5.0 * x) + 1.0) * 0.5
    # define variables and parameters
    @variables temp lday pet prcp snowfall rainfall snowpack melt
    @variables soilwater pet evap baseflow surfaceflow flow rainfall
    @parameters Tmin Tmax Df Smax Qmax f

    # define model components
    bucket_1 = @hydrobucket :surface begin
        fluxes = begin
            @hydroflux begin
                snowfall ~ step_func(Tmin - temp) * prcp
                rainfall ~ step_func(temp - Tmin) * prcp
            end
            @hydroflux melt ~ step_func(temp - Tmax) * step_func(snowpack) * min(snowpack, Df * (temp - Tmax))
            @hydroflux pet ~ 29.8 * lday * 24 * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
        end
        dfluxes = begin
            @stateflux snowpack ~ snowfall - melt
        end
    end

    bucket_2 = @hydrobucket :soil begin
        fluxes = begin
            @hydroflux evap ~ step_func(soilwater) * pet * min(1.0, soilwater / Smax)
            @hydroflux baseflow ~ step_func(soilwater) * Qmax * exp(-f * (max(0.0, Smax - soilwater)))
            @hydroflux surfaceflow ~ max(0.0, soilwater - Smax)
            @hydroflux flow ~ baseflow + surfaceflow
        end
        dfluxes = begin
            @stateflux soilwater ~ (rainfall + melt) - (evap + flow)
        end
    end

    exphydro_model = @hydromodel :exphydro begin
        bucket_1
        bucket_2
    end

    # predefine the parameters
    f, Smax, Qmax, Df, Tmax, Tmin = 0.01674478, 1709.461015, 18.46996175, 2.674548848, 0.175739196, -2.092959084

    # load data
    file_path = "../data/exphydro/01013500.csv"
    data = CSV.File(file_path)
    df = DataFrame(data)
    ts = collect(1:10000)
    lday_vec = df[ts, "dayl(day)"]
    prcp_vec = df[ts, "prcp(mm/day)"]
    temp_vec = df[ts, "tmean(C)"]
    flow_vec = df[ts, "flow(mm)"]

    tunable_pas = ComponentVector(params=ComponentVector(f=f, Smax=Smax, Qmax=Qmax, Df=Df, Tmax=Tmax, Tmin=Tmin))
    const_pas = ComponentVector(initstates=ComponentVector(snowpack=0.0, soilwater=1300.0))

    # parameters optimization
    input = (prcp=prcp_vec, lday=lday_vec, temp=temp_vec)
    input_matrix = Matrix(reduce(hcat, collect(input))')
    output = (flow=flow_vec,)
    config = (solver=ODESolver(), interp=LinearInterpolation)

    ntp_pre = HydroModelTools.NamedTuplePreprocessor(exphydro_model.infos)
    ntp_post = HydroModelTools.NamedTuplePostprocessor(exphydro_model.infos)

    opt_func(i, p; kw...) = begin
        @pipe (i, p) |>
            ntp_pre(_[1], _[2]) |>
            exphydro_model(_[1], _[2]; kw...) |>
            ntp_post(_)
    end

    @testset "HydroOptimizer" begin
        # build optimizer
        hydro_opt = HydroOptimizer(component=opt_func, maxiters=100, solve_alg=BBO_adaptive_de_rand_1_bin_radiuslimited())
        lb_list = [0.0, 100.0, 10.0, 0.0, 0.0, -3.0]
        ub_list = [0.1, 2000.0, 50.0, 5.0, 3.0, 0.0]
        config = (solver=ODESolver(), interp=LinearInterpolation)
        opt_params, loss_df = hydro_opt([input], [output], tunable_pas=tunable_pas, const_pas=const_pas, config=[config], lb=lb_list, ub=ub_list, return_loss_df=true)
        @test true
    end

    @testset "GradOptimizer" begin
        config = (solver=ODESolver(sensealg=GaussAdjoint(autodiff=true)), interp=LinearInterpolation)
        grad_opt = HydroOptimizer(component=opt_func, maxiters=100, solve_alg=Adam(), adtype=AutoForwardDiff())
        opt_params, loss_df = grad_opt([input], [output], tunable_pas=tunable_pas, const_pas=const_pas, config=[config], return_loss_df=true)
        @test true
    end
end