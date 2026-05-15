function synthetic_df()
    return DataFrame(
        firm_id = [1, 1, 2, 2, 3, 3, 1, 2],
        worker_id = [10, 11, 11, 12, 12, 13, 13, 10],
        y = [1.2, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8],
    )
end

function repeated_df()
    return DataFrame(
        firm_id = [1, 1, 1, 2, 2, 3, 3, 1, 2, 2],
        worker_id = [10, 10, 11, 11, 12, 12, 13, 13, 10, 10],
        y = [1.2, 1.1, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8, 0.85],
    )
end

function custom_column_df()
    return DataFrame(
        employer = ["a", "a", "b", "b", "c", "c"],
        person = ["p1", "p2", "p2", "p3", "p3", "p4"],
        outcome = [1.0, missing, 0.8, 1.1, 0.9, 1.2],
    )
end

function fitted_exact(; decompose=false)
    return gmrf_mle(
        synthetic_df();
        solver=ExactCholesky(optim_iters=5, polish=false),
        decompose=decompose,
        seed=1,
        verbose=false,
    )
end
