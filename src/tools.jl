"""
    NamedTuplePreprocessor
"""
@kwdef struct NamedTuplePreprocessor <: AbstractDataPreprocessor
    infos::NamedTuple
end

(pre::NamedTuplePreprocessor)(input::NamedTuple, params) = (permutedims(reduce(hcat, [input[k] for k in get_input_names(pre)])), params)

function (pre::NamedTuplePreprocessor)(input::Vector{<:NamedTuple}, params)
    input_mats = [reduce(hcat, [input[i][k] for k in get_input_names(pre)]) for i in eachindex(input)]
    input_arr = permutedims(reduce((m1, m2) -> cat(m1, m2, dims=3), input_mats), (2, 3, 1))
    return (input_arr, params)
end

"""
    NamedTuplePostprocessor
"""
@kwdef struct NamedTuplePostprocessor <: AbstractDataPostprocessor
    infos::NamedTuple
end

function (post::NamedTuplePostprocessor)(output::AbstractArray{T,2}) where {T}
    output_names_tuple = Tuple(vcat(get_state_names(post), get_output_names(post)))
    return NamedTuple{output_names_tuple}(eachslice(output, dims=1))
end

function (post::NamedTuplePostprocessor)(output::AbstractArray{T,3}) where {T}
    output_names_tuple = Tuple(vcat(get_state_names(post), get_output_names(post)))
    return [NamedTuple{output_names_tuple}(eachslice(output_, dims=1)) for output_ in eachslice(output, dims=2)]
end

"""
    SelectComponentOutlet
"""
@kwdef struct SelectComponentOutlet <: AbstractDataPostprocessor
    infos::NamedTuple
    outlet::Int
end

(post::SelectComponentOutlet)(output::AbstractArray{T,3}) where {T} = output[:, post.outlet, :]