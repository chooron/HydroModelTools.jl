@kwdef struct HydroOptimizer{C,S,A} <: AbstractHydroOptimizer
    component::C
    solve_alg::S
    adtype::A = nothing
    maxiters::Int = 1000
    warmup::Int = 100
    loss_func::Function = (obs, sim) -> sum((obs .- sim) .^ 2) / length(obs)
    callback_func::Function = (loss_recorder) -> get_callback_func(Progress(maxiters, desc="Training..."), loss_recorder)
    objective_func::Function = get_hydro_objective(component, loss_func, warmup)
end

function get_hydro_objective(component, loss_func, warmup)
    #* Constructing the objective function for optimization
    function objective(x::AbstractVector{T}, p) where {T}
        #* Optimization arguments: hydro component, input data, time index, ode solver,
        #*                         tunable parameters axes and default model params
        inputs, targets, run_kwargs, tunable_axes, default_model_pas = p
        #* Use merge_ca to replace the tunable parameters inner the model parameters
        tmp_tunable_pas = ComponentArray(x, tunable_axes)
        tmp_pas = update_ca(default_model_pas, tmp_tunable_pas)
        loss = mean(map(eachindex(inputs)) do i
            tmp_pred = component(inputs[i], tmp_pas; run_kwargs[i]...)
            tmp_loss = mean([loss_func(target[warmup:end], tmp_pred[j][warmup:end]) for (j, target) in enumerate(targets[i])])
            tmp_loss
        end)
        loss
    end
    return objective
end

function (opt::HydroOptimizer{C,S,A})(
    input::Vector,
    target::Vector;
    tunable_pas::ComponentVector,
    const_pas::ComponentVector,
    run_kwargs::Vector=fill(Dict(), length(input)),
    kwargs...
) where {C,S,A}
    loss_recorder = NamedTuple[]
    callback_func = opt.callback_func(loss_recorder)
    tunable_axes = getaxes(tunable_pas)
    default_model_pas = ComponentArray(merge_recursive(NamedTuple(tunable_pas), NamedTuple(const_pas)))
    prob_args = (input, target, run_kwargs, tunable_axes, default_model_pas)
    #* Constructing and solving optimization problems
    @info "The size of tunable parameters is $(length(tunable_pas))"
    if A == Nothing
        optf = Optimization.OptimizationFunction(opt.objective_func)
        lb = get(kwargs, :lb, zeros(length(tunable_pas)))
        ub = get(kwargs, :ub, ones(length(tunable_pas)) .* 100)
        optprob = Optimization.OptimizationProblem(optf, collect(tunable_pas), prob_args, lb=lb, ub=ub)
    else
        optf = Optimization.OptimizationFunction(opt.objective_func, opt.adtype)
        optprob = Optimization.OptimizationProblem(optf, collect(tunable_pas), prob_args)
    end
    sol = Optimization.solve(optprob, opt.solve_alg, callback=callback_func, maxiters=opt.maxiters)
    opt_pas = update_ca(default_model_pas, ComponentVector(sol.u, tunable_axes))
    if get(kwargs, :return_loss_df, false)
        loss_df = DataFrame(loss_recorder)
        return opt_pas, loss_df
    else
        return opt_pas
    end
end
