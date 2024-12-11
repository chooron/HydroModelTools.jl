@testset "test ode solved results" begin
    prcp_itp = LinearInterpolation(input_ntp.prcp, ts)
    temp_itp = LinearInterpolation(input_ntp.temp, ts)

    function snowpack_bucket!(du, u, p, t)
        snowpack_ = u[1]
        Df, Tmax, Tmin = p.Df, p.Tmax, p.Tmin
        prcp_, temp_ = prcp_itp(t), temp_itp(t)
        snowfall_ = step_func(Tmin - temp_) * prcp_
        melt_ = step_func(temp_ - Tmax) * step_func(snowpack_) * min(snowpack_, Df * (temp_ - Tmax))
        du[1] = snowfall_ - melt_
    end
    prob = ODEProblem(snowpack_bucket!, [init_states.snowpack], (ts[1], ts[end]), params)
    sol = solve(prob, Tsit5(), saveat=ts, reltol=1e-3, abstol=1e-3)
    num_u = length(prob.u0)
    manual_result = [sol[i, :] for i in 1:num_u]
    ele_params_idx = [getaxes(pas[:params])[1][nm].idx for nm in HydroModels.get_param_names(snow_ele)]
    paramfunc = (p) -> [p[:params][idx] for idx in ele_params_idx]

    param_func, nn_param_func = HydroModels._get_parameter_extractors(snow_ele, pas)
    itpfunc_list = map((var) -> LinearInterpolation(var, ts, extrapolate=true), eachrow(input))
    ode_input_func = (t) -> [itpfunc(t) for itpfunc in itpfunc_list]
    du_func = HydroModels._get_du_func(snow_ele, ode_input_func, param_func, nn_param_func)
    solver = ManualSolver()
    initstates_mat = collect(pas[:initstates][HydroModels.get_state_names(snow_ele)])
    #* solve the problem by call the solver
    solved_states = solver(du_func, pas, initstates_mat, ts)
    @test manual_result[1] == solved_states[1, :]
end