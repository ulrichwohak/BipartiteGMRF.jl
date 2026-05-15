entity_ref(side::Symbol, id) = (side = side, id = id)

function resolve_firms(problem::GMRFProblem, ids)
    refs = Any[]
    idx = Int[]
    for id in ids
        haskey(problem.firm_to_index, id) ||
            throw(ArgumentError("Firm ID $(id) not found in the fitted problem."))
        i = problem.firm_to_index[id]
        push!(refs, entity_ref(:firm, id))
        push!(idx, i)
    end
    return refs, idx
end

function resolve_workers(problem::GMRFProblem, ids)
    refs = Any[]
    idx = Int[]
    offset = problem.N_firms
    for id in ids
        haskey(problem.worker_to_index, id) ||
            throw(ArgumentError("Worker ID $(id) not found in the fitted problem."))
        i = problem.worker_to_index[id]
        push!(refs, entity_ref(:worker, id))
        push!(idx, offset + i)
    end
    return refs, idx
end

function resolve_entities(problem::GMRFProblem; firms=Any[], workers=Any[])
    f_refs, f_idx = resolve_firms(problem, collect(firms))
    w_refs, w_idx = resolve_workers(problem, collect(workers))
    return vcat(f_refs, w_refs), vcat(f_idx, w_idx)
end

function extract_by_columns(F, n::Int, row_idx::Vector{Int}, col_idx::Vector{Int}; batch_size::Int=16)
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive."))
    out = Matrix{Float64}(undef, length(row_idx), length(col_idx))
    rhs = zeros(Float64, n, min(batch_size, max(length(col_idx), 1)))
    start = 1
    while start <= length(col_idx)
        stop = min(start + batch_size - 1, length(col_idx))
        width = stop - start + 1
        rhs_view = view(rhs, :, 1:width)
        fill!(rhs_view, 0.0)
        @inbounds for j in 1:width
            rhs_view[col_idx[start + j - 1], j] = 1.0
        end
        sol = F \ rhs_view
        out[:, start:stop] .= sol[row_idx, 1:width]
        start = stop + 1
    end
    return out
end

function extract_submatrix(F, n::Int, row_idx::Vector{Int}, col_idx::Vector{Int}; batch_size::Int=16)
    if length(col_idx) <= length(row_idx)
        return extract_by_columns(F, n, row_idx, col_idx; batch_size=batch_size)
    end
    return Matrix(transpose(extract_by_columns(F, n, col_idx, row_idx; batch_size=batch_size)))
end

function cov_block(
    op::CovarianceOperator;
    firms=Any[],
    workers=Any[],
    row_firms=Any[],
    row_workers=Any[],
    col_firms=Any[],
    col_workers=Any[],
    batch_size::Int=16,
)
    problem = op.result.problem
    principal = !isempty(firms) || !isempty(workers)
    rectangular = !isempty(row_firms) || !isempty(row_workers) || !isempty(col_firms) || !isempty(col_workers)
    principal && rectangular &&
        throw(ArgumentError("Use either firms/workers for a principal block or row_/col_ arguments for a rectangular block."))

    if principal
        rows, row_idx = resolve_entities(problem; firms=firms, workers=workers)
        cols, col_idx = rows, row_idx
    else
        isempty(row_firms) && isempty(row_workers) && isempty(col_firms) && isempty(col_workers) &&
            throw(ArgumentError("No covariance block requested."))
        if isempty(row_firms) && isempty(row_workers)
            row_firms = col_firms
            row_workers = col_workers
        end
        if isempty(col_firms) && isempty(col_workers)
            col_firms = row_firms
            col_workers = row_workers
        end
        rows, row_idx = resolve_entities(problem; firms=row_firms, workers=row_workers)
        cols, col_idx = resolve_entities(problem; firms=col_firms, workers=col_workers)
    end

    isempty(row_idx) && throw(ArgumentError("Requested row set is empty."))
    isempty(col_idx) && throw(ArgumentError("Requested column set is empty."))
    n = problem.N_firms + problem.N_workers
    mat = extract_submatrix(op.factor, n, row_idx, col_idx; batch_size=batch_size)
    op.units == :original && (mat .*= problem.y_std^2)
    return CovarianceBlock(mat, rows, cols, op.kind, op.units)
end
