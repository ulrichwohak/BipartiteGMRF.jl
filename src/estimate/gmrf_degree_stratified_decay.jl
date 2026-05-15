#!/usr/bin/env julia

module GMRFDegreeStratifiedDecay

const PROJECT_ROOT = realpath(normpath(joinpath(@__DIR__, "..", "..")))

using CSV
using JSON
using Parquet2
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using Printf: @printf

include("gmrf_outcome_cov_distance_decay.jl")

const OutcomeDecay = GMRFOutcomeCovDistanceDecay
const Decay = OutcomeDecay.Decay
const CovTool = OutcomeDecay.CovTool

struct DegreeBin
    name::Symbol
    min_degree::Int
    max_degree::Union{Int,Nothing}
end

struct SecondShortestSettings
    mode::Symbol
    eps::Float64
    threshold_source::String
    threshold_base::Union{Int,Nothing}
end

default_estimates_path() = joinpath(
    PROJECT_ROOT,
    "output/gmrfmle/full/mincomp-2-2/estimates.txt",
)

default_degree_bins() = DegreeBin[
    DegreeBin(:low, 1, 2),
    DegreeBin(:medium, 3, 5),
    DegreeBin(:high, 6, nothing),
]

function parse_degree_bins(raw::String)::Vector{DegreeBin}
    bins = DegreeBin[]
    for token in CovTool.split_csv_arg(raw)
        occursin("=", token) || error("Degree bin '$token' must have form name=min:max")
        name_raw, range_raw = split(token, "=", limit=2)
        bounds = split(range_raw, ":", limit=2)
        length(bounds) == 2 || error("Degree bin '$token' must have form name=min:max")
        lo = parse(Int, strip(bounds[1]))
        hi_raw = lowercase(strip(bounds[2]))
        hi = hi_raw in ("inf", "infinity", "+inf") ? nothing : parse(Int, hi_raw)
        lo >= 0 || error("Degree bin lower bound must be nonnegative: $token")
        (hi === nothing || hi >= lo) || error("Degree bin upper bound must be >= lower bound: $token")
        push!(bins, DegreeBin(Symbol(strip(name_raw)), lo, hi))
    end
    isempty(bins) && error("At least one degree bin is required")
    return bins
end

bin_label(bin::DegreeBin) = String(bin.name)

function degree_in_bin(degree::Integer, bin::DegreeBin)::Bool
    degree >= bin.min_degree && (bin.max_degree === nothing || degree <= bin.max_degree)
end

function bin_spec(bin::DegreeBin)::String
    hi = bin.max_degree === nothing ? "Inf" : string(bin.max_degree)
    return "$(bin.min_degree):$(hi)"
end

function parse_objects(raw::String)::Vector{Symbol}
    s = lowercase(strip(raw))
    s == "both" && return [:latent, :outcome]
    objects = Symbol[]
    for token in CovTool.split_csv_arg(raw)
        obj = Symbol(lowercase(token))
        obj in (:latent, :outcome) || error("--objects must be latent, outcome, or both; got $token")
        obj in objects || push!(objects, obj)
    end
    isempty(objects) && error("--objects must select at least one object")
    return objects
end

function parse_second_shortest_mode(raw::String)::Symbol
    mode = Symbol(lowercase(strip(raw)))
    mode in (:filter, :annotate, :off) ||
        error("--second-shortest must be filter, annotate, or off; got $(raw)")
    return mode
end

function parse_bucket_scope(raw::String)::Symbol
    scope = Symbol(lowercase(strip(raw)))
    scope in (:endpoint, :path) ||
        error("--bucket-scope must be endpoint or path; got $(raw)")
    return scope
end

function parse_anchor_selection(raw::String)::Symbol
    selection = Symbol(lowercase(strip(raw)))
    selection in (:coverage, :degree) ||
        error("--anchor-selection must be coverage or degree; got $(raw)")
    return selection
end

function parse_isolation_eps(raw::String)::Float64
    eps = parse(Float64, strip(raw))
    0.0 < eps < 1.0 || error("--isolation-eps must be in (0, 1); got $(raw)")
    return eps
end

function auto_isolation_threshold_base(rho::Real, eps::Real)::Int
    absrho = abs(Float64(rho))
    eps64 = Float64(eps)
    0.0 < eps64 < 1.0 || error("--isolation-eps must be in (0, 1); got $(eps)")
    absrho == 0.0 && return 1
    absrho < 1.0 ||
        error("Automatic isolation threshold requires abs(rho) < 1; got rho=$(rho). Use --isolation-threshold=N to override.")
    return max(1, ceil(Int, log(eps64) / log(absrho)))
end

function parse_isolation_threshold(raw::String, rho::Real, eps::Real)::Tuple{String,Int}
    s = lowercase(strip(raw))
    if s == "auto"
        return ("auto", auto_isolation_threshold_base(rho, eps))
    end

    threshold = parse(Int, s)
    threshold >= 1 || error("--isolation-threshold must be auto or an integer >= 1; got $(raw)")
    return ("manual", threshold)
end

function second_shortest_settings(mode::Symbol,
                                  eps::Float64,
                                  threshold_source::String,
                                  threshold_base::Union{Int,Nothing})::SecondShortestSettings
    mode == :off && return SecondShortestSettings(mode, eps, "off", nothing)
    threshold_base === nothing && error("Second-shortest mode $(mode) requires an isolation threshold")
    return SecondShortestSettings(mode, eps, threshold_source, threshold_base)
end

function parse_flags(args::Vector{String})
    positional = filter(arg -> !startswith(arg, "--"), args)
    flags_list = filter(arg -> startswith(arg, "--"), args)

    flags = Dict{String,String}()
    for flag in flags_list
        occursin("=", flag) || error("Unknown flag: $(flag). Flags must be --key=value.")
        key, value = split(flag, "=", limit=2)
        flags[key] = value
    end

    estimates_path = isempty(positional) ? default_estimates_path() : positional[1]
    length(positional) <= 1 || error("Expected at most one positional estimates path")
    return (estimates_path=estimates_path, flags=flags)
end

function node_side(prep, latent_index::Int)::Symbol
    return latent_index <= prep.N_F ? :firm : :person
end

function opposite_side(side::Symbol)::Symbol
    side == :firm && return :person
    side == :person && return :firm
    error("Unknown side: $side")
end

function side_pair_label(left::Symbol, right::Symbol)::String
    return string(String(left), "-", String(right))
end

function local_index(prep, side::Symbol, latent_index::Int)::Int
    return side == :firm ? latent_index : latent_index - prep.N_F
end

function node_degree(prep, latent_index::Int)::Int
    side = node_side(prep, latent_index)
    idx = local_index(prep, side, latent_index)
    d = side == :firm ? prep.d_f[idx] : prep.d_m[idx]
    return Int(round(d))
end

function entity_label(prepared, latent_index::Int)::String
    side = node_side(prepared.prep, latent_index)
    return CovTool.entity_label(Decay.entity_ref(prepared, side, latent_index))
end

function entity_id_string(prepared, latent_index::Int)::String
    side = node_side(prepared.prep, latent_index)
    return string(Decay.entity_id(prepared, side, latent_index))
end

function path_degrees(prep, path::Vector{Int})::Vector{Int}
    return [node_degree(prep, node) for node in path]
end

function path_is_tier_pure(prep, path::Vector{Int}, bin::DegreeBin)::Bool
    return all(degree_in_bin(node_degree(prep, node), bin) for node in path)
end

function path_allowed_by_bucket_scope(prep, path::Vector{Int}, bin::DegreeBin, bucket_scope::Symbol)::Bool
    bucket_scope == :endpoint && return degree_in_bin(node_degree(prep, first(path)), bin) &&
        degree_in_bin(node_degree(prep, last(path)), bin)
    bucket_scope == :path && return path_is_tier_pure(prep, path, bin)
    error("Unknown bucket_scope: $(bucket_scope)")
end

function edge_weight(prep, u::Int, v::Int)::Float64
    su = node_side(prep, u)
    sv = node_side(prep, v)
    su != sv || error("Adjacent bipartite nodes must be on opposite sides: $u, $v")

    if su == :firm
        f = u
        p = v - prep.N_F
    else
        f = v
        p = u - prep.N_F
    end

    w = prep.A_fm[f, p]
    w != 0 || error("Path edge not present in A_fm: $u, $v")
    return Float64(prep.df_is[f] * w * prep.dm_is[p])
end

function path_weight(prep, path::Vector{Int})::Float64
    length(path) <= 1 && return 1.0
    weight = 1.0
    for i in 1:(length(path) - 1)
        weight *= edge_weight(prep, path[i], path[i + 1])
    end
    return weight
end

mutable struct YenWorkspace
    visit_token::Int
    seen::Vector{Int}
    parent::Vector{Int}
    dist::Vector{Int}
    queue::Vector{Int}
end

function YenWorkspace(n::Int)
    return YenWorkspace(0, zeros(Int, n), zeros(Int, n), zeros(Int, n), Vector{Int}(undef, n))
end

edge_key(u::Int, v::Int) = u < v ? (u, v) : (v, u)

function next_yen_token!(workspace::YenWorkspace)
    workspace.visit_token += 1
    if workspace.visit_token == typemax(Int)
        fill!(workspace.seen, 0)
        workspace.visit_token = 1
    end
    return workspace.visit_token
end

function bfs_path_avoiding(neighbors::Vector{Vector{Int}},
                           start::Int,
                           target::Int,
                           banned_vertices::Set{Int},
                           banned_edges::Set{Tuple{Int,Int}},
                           workspace::YenWorkspace)::Union{Vector{Int},Nothing}
    start in banned_vertices && return nothing
    target in banned_vertices && return nothing

    token = next_yen_token!(workspace)
    head = 1
    tail = 1
    workspace.queue[tail] = start
    workspace.seen[start] = token
    workspace.parent[start] = 0
    workspace.dist[start] = 0

    while head <= tail
        u = workspace.queue[head]
        head += 1
        if u == target
            path = Int[]
            node = target
            while node != 0
                push!(path, node)
                node = workspace.parent[node]
            end
            reverse!(path)
            return path
        end

        for v in neighbors[u]
            workspace.seen[v] == token && continue
            v in banned_vertices && continue
            edge_key(u, v) in banned_edges && continue
            tail += 1
            workspace.queue[tail] = v
            workspace.seen[v] = token
            workspace.parent[v] = u
            workspace.dist[v] = workspace.dist[u] + 1
        end
    end

    return nothing
end

function yen_second_shortest_path_length(neighbors::Vector{Vector{Int}},
                                         path::Vector{Int},
                                         workspace::YenWorkspace=YenWorkspace(length(neighbors)))::Union{Int,Nothing}
    length(path) >= 2 || return nothing
    start = first(path)
    target = last(path)
    t1 = length(path) - 1
    shortest = bfs_path_avoiding(
        neighbors,
        start,
        target,
        Set{Int}(),
        Set{Tuple{Int,Int}}(),
        workspace,
    )
    shortest === nothing && error("Yen found no path between selected endpoints $(start) and $(target)")
    length(shortest) - 1 == t1 ||
        error("Yen shortest distance $(length(shortest) - 1) does not match selected path distance $(t1) for $(start) -> $(target)")

    best = typemax(Int)
    for spur_pos in 1:t1
        spur = path[spur_pos]
        banned_vertices = Set(path[1:(spur_pos - 1)])
        banned_edges = Set{Tuple{Int,Int}}([edge_key(path[spur_pos], path[spur_pos + 1])])
        spur_path = bfs_path_avoiding(
            neighbors,
            spur,
            target,
            banned_vertices,
            banned_edges,
            workspace,
        )
        spur_path === nothing && continue
        candidate_distance = (spur_pos - 1) + length(spur_path) - 1
        candidate_distance < best && (best = candidate_distance)
    end

    return best == typemax(Int) ? nothing : best
end

function second_path_status(t1::Int, t2::Union{Int,Nothing})::String
    t2 === nothing && return "none_within_threshold"
    t2 == t1 && return "equal_shortest_alternate"
    t2 < t1 && return "shorter_alternate"
    return "short_alternate"
end

function second_path_metadata(neighbors::Union{Nothing,Vector{Vector{Int}}},
                              path::Vector{Int},
                              rho::Float64,
                              settings::SecondShortestSettings,
                              workspace::Union{Nothing,YenWorkspace}=nothing)
    t1 = length(path) - 1
    if settings.mode == :off
        return (
            second_shortest_mode="off",
            second_shortest_distance=missing,
            second_shortest_censored_at=missing,
            second_shortest_pass=true,
            isolation_threshold=missing,
            rho_second_power=missing,
            rho_gap_power=missing,
            alternate_path_status="not_checked",
        )
    end

    neighbors === nothing && error("Second-shortest mode $(settings.mode) requires graph neighbors")
    workspace === nothing && error("Second-shortest mode $(settings.mode) requires a YenWorkspace")
    threshold = max(t1, settings.threshold_base::Int)
    t2 = yen_second_shortest_path_length(neighbors, path, workspace)
    pass = t2 === nothing || t2 > threshold

    return (
        second_shortest_mode=String(settings.mode),
        second_shortest_distance=t2 === nothing ? missing : t2,
        second_shortest_censored_at=missing,
        second_shortest_pass=pass,
        isolation_threshold=threshold,
        rho_second_power=t2 === nothing ? missing : rho^t2,
        rho_gap_power=t2 === nothing ? missing : rho^(t2 - t1),
        alternate_path_status=second_path_status(t1, t2),
    )
end

function path_metadata(prepared, path::Vector{Int}, bin::DegreeBin, rho::Float64)
    prep = prepared.prep
    d = length(path) - 1
    degrees = path_degrees(prep, path)
    weight = path_weight(prep, path)
    rho_power = rho^d
    return (
        tier=bin_label(bin),
        distance=d,
        path_latent_indices=join(string.(path), " "),
        path_labels=join((entity_label(prepared, node) for node in path), " -> "),
        path_degrees=join(string.(degrees), " "),
        path_min_degree=minimum(degrees),
        path_mean_degree=mean(degrees),
        path_max_degree=maximum(degrees),
        path_weight=weight,
        rho_power=rho_power,
        weighted_benchmark=rho_power * weight,
    )
end

function sorted_tier_nodes(prepared, bin::DegreeBin, side::Symbol)::Vector{Int}
    prep = prepared.prep
    nodes = collect(Decay.side_range(prep, side))
    filter!(node -> degree_in_bin(node_degree(prep, node), bin), nodes)
    sort!(nodes; by = node -> (
        -node_degree(prep, node),
        entity_id_string(prepared, node),
        node,
    ))
    return nodes
end

function anchor_candidate_count(total::Int,
                                anchors_per_tier_side::Int,
                                candidate_multiplier::Int,
                                candidate_max::Int)::Int
    candidate_multiplier >= 1 || error("--anchor-candidate-multiplier must be >= 1")
    candidate_max >= 1 || error("--anchor-candidate-max must be >= 1")
    desired = max(anchors_per_tier_side, anchors_per_tier_side * candidate_multiplier)
    return min(total, desired, candidate_max)
end

function anchor_distance_score(prepared,
                               neighbors::Vector{Vector{Int}},
                               anchor::Int,
                               targets_by_distance::Dict{Int,Vector{Int}},
                               bin::DegreeBin,
                               bucket_scope::Symbol,
                               max_distance::Int)::Int
    isempty(targets_by_distance) && return 0
    dist, parent = OutcomeDecay.bfs_tree(neighbors, anchor; max_distance=max_distance)
    score = 0
    for (d, targets) in targets_by_distance
        found = false
        for target in targets
            target == anchor && continue
            dist[target] == d || continue
            if bucket_scope == :path
                path = OutcomeDecay.reconstruct_path(parent, anchor, target)
                path_is_tier_pure(prepared.prep, path, bin) || continue
            end
            found = true
            break
        end
        score += found ? 1 : 0
    end
    return score
end

function choose_anchors(prepared,
                        neighbors::Vector{Vector{Int}},
                        anchors_all::Vector{Int},
                        targets_by_distance::Dict{Int,Vector{Int}},
                        bin::DegreeBin;
                        anchors_per_tier_side::Int,
                        max_distance::Int,
                        bucket_scope::Symbol,
                        anchor_selection::Symbol,
                        anchor_candidate_multiplier::Int,
                        anchor_candidate_max::Int)::Vector{Int}
    isempty(anchors_all) && return Int[]
    if anchor_selection == :degree
        return first(anchors_all, min(anchors_per_tier_side, length(anchors_all)))
    elseif anchor_selection != :coverage
        error("Unknown anchor_selection: $(anchor_selection)")
    end

    n_candidates = anchor_candidate_count(length(anchors_all),
                                          anchors_per_tier_side,
                                          anchor_candidate_multiplier,
                                          anchor_candidate_max)
    candidates = collect(first(anchors_all, n_candidates))
    sort!(candidates; by = anchor -> (
        -anchor_distance_score(prepared, neighbors, anchor, targets_by_distance,
                               bin, bucket_scope, max_distance),
        -node_degree(prepared.prep, anchor),
        entity_id_string(prepared, anchor),
        anchor,
    ))
    return first(candidates, min(anchors_per_tier_side, length(candidates)))
end

function selected_path_row(object::Symbol,
                           bin::DegreeBin,
                           endpoint_group::String,
                           anchor_side::Symbol,
                           path::Vector{Int},
                           prepared,
                           rho::Float64;
                           outcome=nothing,
                           isolation=nothing)
    meta = path_metadata(prepared, path, bin, rho)
    isolation_meta = isolation === nothing ? (
        second_shortest_mode="off",
        second_shortest_distance=missing,
        second_shortest_censored_at=missing,
        second_shortest_pass=true,
        isolation_threshold=missing,
        rho_second_power=missing,
        rho_gap_power=missing,
        alternate_path_status="not_checked",
    ) : isolation
    anchor = first(path)
    target = last(path)

    if outcome === nothing
        left_firm_latent = missing
        left_person_latent = missing
        right_firm_latent = missing
        right_person_latent = missing
        left_firm_id = missing
        left_person_id = missing
        right_firm_id = missing
        right_person_id = missing
    else
        left_firm_latent = outcome.left_firm_latent
        left_person_latent = outcome.left_person_latent
        right_firm_latent = outcome.right_firm_latent
        right_person_latent = outcome.right_person_latent
        left_firm_id = outcome.left_firm_id
        left_person_id = outcome.left_person_id
        right_firm_id = outcome.right_firm_id
        right_person_id = outcome.right_person_id
    end

    return merge(meta, isolation_meta, (
        object=String(object),
        endpoint_group=endpoint_group,
        anchor_side=String(anchor_side),
        anchor_id=entity_id_string(prepared, anchor),
        target_id=entity_id_string(prepared, target),
        anchor_latent=anchor,
        target_latent=target,
        anchor_degree=node_degree(prepared.prep, anchor),
        target_degree=node_degree(prepared.prep, target),
        left_firm_latent=left_firm_latent,
        left_person_latent=left_person_latent,
        right_firm_latent=right_firm_latent,
        right_person_latent=right_person_latent,
        left_firm_id=left_firm_id,
        left_person_id=left_person_id,
        right_firm_id=right_firm_id,
        right_person_id=right_person_id,
    ))
end

function coverage_row(object::Symbol,
                      bin::DegreeBin,
                      endpoint_group::String,
                      anchor_side::Symbol,
                      distance::Int,
                      requested::Int,
                      anchors_considered::Int,
                      available::Int,
                      tier_pure_available::Int,
                      second_path_checked::Int,
                      second_path_rejected::Int,
                      selected::Int)
    return (
        object=String(object),
        tier=bin_label(bin),
        endpoint_group=endpoint_group,
        anchor_side=String(anchor_side),
        distance=distance,
        requested=requested,
        anchors_considered=anchors_considered,
        available=available,
        tier_pure_available=tier_pure_available,
        second_path_checked=second_path_checked,
        second_path_rejected=second_path_rejected,
        selected=selected,
    )
end

function select_latent_paths(prepared,
                             neighbors::Vector{Vector{Int}},
                             bins::Vector{DegreeBin};
                             sides::Vector{Symbol},
                             anchors_per_tier_side::Int,
                             per_cell::Int,
                             max_distance::Int,
                             rho::Float64,
                             second_shortest_settings::SecondShortestSettings=SecondShortestSettings(:off, 0.01, "off", nothing),
                             bucket_scope::Symbol=:endpoint,
                             anchor_selection::Symbol=:coverage,
                             anchor_candidate_multiplier::Int=25,
                             anchor_candidate_max::Int=250)
    records = NamedTuple[]
    coverage = NamedTuple[]
    seen_pairs = Set{Tuple{Int,Int}}()
    candidate_cache = Dict{Tuple{Symbol,Symbol},Vector{Int}}()
    yen_workspace = second_shortest_settings.mode == :off ? nothing : YenWorkspace(length(neighbors))

    for bin in bins, anchor_side in sides
        anchors_all = get!(candidate_cache, (bin.name, anchor_side)) do
            sorted_tier_nodes(prepared, bin, anchor_side)
        end
        targets_by_distance = Dict{Int,Vector{Int}}()
        for d in 1:max_distance
            target_side = iseven(d) ? anchor_side : opposite_side(anchor_side)
            targets_by_distance[d] = get!(candidate_cache, (bin.name, target_side)) do
                sorted_tier_nodes(prepared, bin, target_side)
            end
        end
        anchors = choose_anchors(
            prepared,
            neighbors,
            anchors_all,
            targets_by_distance,
            bin;
            anchors_per_tier_side=anchors_per_tier_side,
            max_distance=max_distance,
            bucket_scope=bucket_scope,
            anchor_selection=anchor_selection,
            anchor_candidate_multiplier=anchor_candidate_multiplier,
            anchor_candidate_max=anchor_candidate_max,
        )

        for d in 1:max_distance
            target_side = iseven(d) ? anchor_side : opposite_side(anchor_side)
            targets = targets_by_distance[d]
            endpoint_group = side_pair_label(anchor_side, target_side)
            available = 0
            tier_pure_available = 0
            second_path_checked = 0
            second_path_rejected = 0
            selected = 0
            preselected = NamedTuple[]
            preselected_keys = Set{Tuple{Int,Int}}()

            for anchor in anchors
                length(preselected) >= per_cell && break
                dist, parent = OutcomeDecay.bfs_tree(neighbors, anchor; max_distance=d)

                for target in targets
                    length(preselected) >= per_cell && break
                    target == anchor && continue
                    dist[target] == d || continue
                    key = (min(anchor, target), max(anchor, target))
                    key in seen_pairs && continue
                    key in preselected_keys && continue
                    path = OutcomeDecay.reconstruct_path(parent, anchor, target)
                    is_tier_pure = path_is_tier_pure(prepared.prep, path, bin)
                    is_tier_pure && (tier_pure_available += 1)
                    path_allowed_by_bucket_scope(prepared.prep, path, bin, bucket_scope) || continue

                    available += 1
                    push!(preselected, (path=path, key=key))
                    push!(preselected_keys, key)
                end
            end

            for candidate in preselected
                isolation = second_path_metadata(
                    second_shortest_settings.mode == :off ? nothing : neighbors,
                    candidate.path,
                    rho,
                    second_shortest_settings,
                    yen_workspace,
                )
                if second_shortest_settings.mode != :off
                    second_path_checked += 1
                end
                if second_shortest_settings.mode == :filter && !isolation.second_shortest_pass
                    second_path_rejected += 1
                    continue
                end

                push!(records, selected_path_row(
                    :latent,
                    bin,
                    endpoint_group,
                    anchor_side,
                    candidate.path,
                    prepared,
                    rho,
                    isolation=isolation,
                ))
                push!(seen_pairs, candidate.key)
                selected += 1
            end

            push!(coverage, coverage_row(
                :latent,
                bin,
                endpoint_group,
                anchor_side,
                d,
                per_cell,
                length(anchors),
                available,
                tier_pure_available,
                second_path_checked,
                second_path_rejected,
                selected,
            ))
        end
    end

    return records, coverage
end

function select_outcome_paths(prepared,
                              neighbors::Vector{Vector{Int}},
                              bins::Vector{DegreeBin};
                              endpoint_sides::Vector{Symbol},
                              anchors_per_tier_side::Int,
                              per_cell::Int,
                              max_distance::Int,
                              rho::Float64,
                              second_shortest_settings::SecondShortestSettings=SecondShortestSettings(:off, 0.01, "off", nothing),
                              bucket_scope::Symbol=:endpoint,
                              anchor_selection::Symbol=:coverage,
                              anchor_candidate_multiplier::Int=25,
                              anchor_candidate_max::Int=250)
    max_distance >= 2 || error("Outcome distance diagnostic requires --max-distance >= 2")
    prep = prepared.prep
    edge_lookup = OutcomeDecay.build_edge_lookup(prep)
    records = NamedTuple[]
    coverage = NamedTuple[]
    seen_pairs = Set{Tuple{Symbol,Int,Int}}()
    candidate_cache = Dict{Tuple{Symbol,Symbol},Vector{Int}}()
    yen_workspace = second_shortest_settings.mode == :off ? nothing : YenWorkspace(length(neighbors))

    for bin in bins, endpoint_side in endpoint_sides
        anchors_all = get!(candidate_cache, (bin.name, endpoint_side)) do
            sorted_tier_nodes(prepared, bin, endpoint_side)
        end
        targets_by_distance = Dict{Int,Vector{Int}}(d => anchors_all for d in 2:2:max_distance)
        anchors = choose_anchors(
            prepared,
            neighbors,
            anchors_all,
            targets_by_distance,
            bin;
            anchors_per_tier_side=anchors_per_tier_side,
            max_distance=max_distance,
            bucket_scope=bucket_scope,
            anchor_selection=anchor_selection,
            anchor_candidate_multiplier=anchor_candidate_multiplier,
            anchor_candidate_max=anchor_candidate_max,
        )
        endpoint_group = string(String(endpoint_side), "-outcome")

        for d in 2:2:max_distance
            targets = anchors_all
            available = 0
            tier_pure_available = 0
            second_path_checked = 0
            second_path_rejected = 0
            selected = 0
            preselected = NamedTuple[]
            preselected_keys = Set{Tuple{Symbol,Int,Int}}()

            for anchor in anchors
                length(preselected) >= per_cell && break
                dist, parent = OutcomeDecay.bfs_tree(neighbors, anchor; max_distance=d)

                for target in targets
                    length(preselected) >= per_cell && break
                    target == anchor && continue
                    dist[target] == d || continue
                    key = (endpoint_side, min(anchor, target), max(anchor, target))
                    key in seen_pairs && continue
                    key in preselected_keys && continue
                    path = OutcomeDecay.reconstruct_path(parent, anchor, target)
                    is_tier_pure = path_is_tier_pure(prep, path, bin)
                    is_tier_pure && (tier_pure_available += 1)
                    path_allowed_by_bucket_scope(prep, path, bin, bucket_scope) || continue

                    available += 1
                    push!(preselected, (path=path, key=key))
                    push!(preselected_keys, key)
                end
            end

            for candidate in preselected
                isolation = second_path_metadata(
                    second_shortest_settings.mode == :off ? nothing : neighbors,
                    candidate.path,
                    rho,
                    second_shortest_settings,
                    yen_workspace,
                )
                if second_shortest_settings.mode != :off
                    second_path_checked += 1
                end
                if second_shortest_settings.mode == :filter && !isolation.second_shortest_pass
                    second_path_rejected += 1
                    continue
                end

                outcome = OutcomeDecay.outcome_path_from_latents(prepared, candidate.path, edge_lookup)
                push!(records, selected_path_row(
                    :outcome,
                    bin,
                    endpoint_group,
                    endpoint_side,
                    candidate.path,
                    prepared,
                    rho;
                    outcome=outcome,
                    isolation=isolation,
                ))
                push!(seen_pairs, candidate.key)
                selected += 1
            end

            push!(coverage, coverage_row(
                :outcome,
                bin,
                endpoint_group,
                endpoint_side,
                d,
                per_cell,
                length(anchors),
                available,
                tier_pure_available,
                second_path_checked,
                second_path_rejected,
                selected,
            ))
        end
    end

    return records, coverage
end

series_key(row) = (String(row.object), String(row.tier), String(row.endpoint_group))

function series_distance_counts(records::Vector{<:NamedTuple})
    distances = Dict{Tuple{String,String,String},Set{Int}}()
    for r in records
        key = series_key(r)
        push!(get!(distances, key, Set{Int}()), Int(r.distance))
    end
    return Dict(key => length(vals) for (key, vals) in distances)
end

function filter_records_for_decay(records::Vector{<:NamedTuple},
                                  coverage::Vector{<:NamedTuple};
                                  min_distances_per_series::Int)
    min_distances_per_series >= 1 || error("--min-distances-per-series must be >= 1")
    counts = series_distance_counts(records)
    keep_keys = Set(key for (key, count) in counts if count >= min_distances_per_series)
    filtered_records = [r for r in records if series_key(r) in keep_keys]

    annotated_coverage = NamedTuple[]
    for c in coverage
        key = series_key(c)
        distance_count = get(counts, key, 0)
        keep = key in keep_keys
        selected_before = Int(c.selected)
        rejected = keep ? 0 : selected_before
        push!(annotated_coverage, merge(c, (
            selected_before_distance_filter=selected_before,
            series_distance_count=distance_count,
            distance_filter_rejected=rejected,
            selected=keep ? selected_before : 0,
        )))
    end

    return filtered_records, annotated_coverage
end

function latent_indices_for_records(records::Vector{<:NamedTuple})::Vector{Int}
    idx = Int[]
    for r in records
        if r.object == "latent"
            push!(idx, r.anchor_latent)
            push!(idx, r.target_latent)
        else
            push!(idx, r.left_firm_latent)
            push!(idx, r.left_person_latent)
            push!(idx, r.right_firm_latent)
            push!(idx, r.right_person_latent)
        end
    end
    sort!(unique!(idx))
    return idx
end

function extract_sigma_blocks(prep, meta, latent_indices::Vector{Int};
                              matrix_mode::Symbol,
                              units::Symbol,
                              batch_size::Int)
    built = CovTool.build_exact_matrices(prep, meta)
    n = prep.N_F + prep.N_M
    scale = units == :original ? prep.y_std^2 : 1.0
    blocks = Dict{Symbol,Matrix{Float64}}()

    for mode in Decay.matrix_modes(matrix_mode)
        if mode == :prior
            @printf("Factoring prior precision Q...\n")
            F = cholesky(Symmetric(built.Q))
        else
            @printf("Factoring posterior precision M = Q + lambda * V'V...\n")
            F = cholesky(Symmetric(built.M))
        end

        @printf("Extracting %s Sigma block for %d latent nodes...\n", String(mode), length(latent_indices))
        blocks[mode] = CovTool.extract_submatrix(F, n, latent_indices, latent_indices; batch_size=batch_size) .* scale
    end

    return blocks
end

function latent_pairwise_dataframe(matrix_name::Symbol,
                                   sigma_block::Matrix{Float64},
                                   latent_indices::Vector{Int},
                                   records::Vector{<:NamedTuple})
    pos = Dict(latent => i for (i, latent) in enumerate(latent_indices))
    rows = NamedTuple[]
    cov_at(i::Int, j::Int) = sigma_block[pos[i], pos[j]]

    for r in records
        r.object == "latent" || continue
        cov = cov_at(r.anchor_latent, r.target_latent)
        var_left = cov_at(r.anchor_latent, r.anchor_latent)
        var_right = cov_at(r.target_latent, r.target_latent)
        denom = var_left > 0 && var_right > 0 ? sqrt(var_left * var_right) : NaN
        corr = isfinite(denom) && denom > 0 ? cov / denom : NaN
        ratio = r.weighted_benchmark != 0 ? corr / r.weighted_benchmark : NaN

        push!(rows, (
            object="latent",
            matrix=String(matrix_name),
            tier=r.tier,
            endpoint_group=r.endpoint_group,
            anchor_side=r.anchor_side,
            distance=r.distance,
            anchor_id=r.anchor_id,
            target_id=r.target_id,
            anchor_latent=r.anchor_latent,
            target_latent=r.target_latent,
            anchor_degree=r.anchor_degree,
            target_degree=r.target_degree,
            cov=cov,
            corr=corr,
            var_left=var_left,
            var_right=var_right,
            rho_power=r.rho_power,
            path_weight=r.path_weight,
            weighted_benchmark=r.weighted_benchmark,
            corr_to_weighted_benchmark=ratio,
            second_shortest_mode=r.second_shortest_mode,
            second_shortest_distance=r.second_shortest_distance,
            second_shortest_censored_at=r.second_shortest_censored_at,
            second_shortest_pass=r.second_shortest_pass,
            isolation_threshold=r.isolation_threshold,
            rho_second_power=r.rho_second_power,
            rho_gap_power=r.rho_gap_power,
            alternate_path_status=r.alternate_path_status,
            path_min_degree=r.path_min_degree,
            path_mean_degree=r.path_mean_degree,
            path_max_degree=r.path_max_degree,
            path_labels=r.path_labels,
            path_degrees=r.path_degrees,
        ))
    end

    return DataFrame(rows)
end

function outcome_pairwise_dataframe(matrix_name::Symbol,
                                    sigma_block::Matrix{Float64},
                                    latent_indices::Vector{Int},
                                    records::Vector{<:NamedTuple})
    pos = Dict(latent => i for (i, latent) in enumerate(latent_indices))
    rows = NamedTuple[]
    cov_at(i::Int, j::Int) = sigma_block[pos[i], pos[j]]

    for r in records
        r.object == "outcome" || continue
        cov_ff = cov_at(r.left_firm_latent, r.right_firm_latent)
        cov_fp = cov_at(r.left_firm_latent, r.right_person_latent)
        cov_pf = cov_at(r.left_person_latent, r.right_firm_latent)
        cov_pp = cov_at(r.left_person_latent, r.right_person_latent)
        cov = cov_ff + cov_fp + cov_pf + cov_pp

        var_left = cov_at(r.left_firm_latent, r.left_firm_latent) +
                   cov_at(r.left_firm_latent, r.left_person_latent) +
                   cov_at(r.left_person_latent, r.left_firm_latent) +
                   cov_at(r.left_person_latent, r.left_person_latent)
        var_right = cov_at(r.right_firm_latent, r.right_firm_latent) +
                    cov_at(r.right_firm_latent, r.right_person_latent) +
                    cov_at(r.right_person_latent, r.right_firm_latent) +
                    cov_at(r.right_person_latent, r.right_person_latent)
        denom = var_left > 0 && var_right > 0 ? sqrt(var_left * var_right) : NaN
        corr = isfinite(denom) && denom > 0 ? cov / denom : NaN
        ratio = r.weighted_benchmark != 0 ? corr / r.weighted_benchmark : NaN

        push!(rows, (
            object="outcome",
            matrix=String(matrix_name),
            tier=r.tier,
            endpoint_group=r.endpoint_group,
            anchor_side=r.anchor_side,
            distance=r.distance,
            anchor_id=r.anchor_id,
            target_id=r.target_id,
            left_outcome="firm:$(r.left_firm_id)+person:$(r.left_person_id)",
            right_outcome="firm:$(r.right_firm_id)+person:$(r.right_person_id)",
            left_firm_id=r.left_firm_id,
            left_person_id=r.left_person_id,
            right_firm_id=r.right_firm_id,
            right_person_id=r.right_person_id,
            left_firm_latent=r.left_firm_latent,
            left_person_latent=r.left_person_latent,
            right_firm_latent=r.right_firm_latent,
            right_person_latent=r.right_person_latent,
            cov=cov,
            corr=corr,
            var_left=var_left,
            var_right=var_right,
            cov_leftfirm_rightfirm=cov_ff,
            cov_leftfirm_rightperson=cov_fp,
            cov_leftperson_rightfirm=cov_pf,
            cov_leftperson_rightperson=cov_pp,
            rho_power=r.rho_power,
            path_weight=r.path_weight,
            weighted_benchmark=r.weighted_benchmark,
            corr_to_weighted_benchmark=ratio,
            second_shortest_mode=r.second_shortest_mode,
            second_shortest_distance=r.second_shortest_distance,
            second_shortest_censored_at=r.second_shortest_censored_at,
            second_shortest_pass=r.second_shortest_pass,
            isolation_threshold=r.isolation_threshold,
            rho_second_power=r.rho_second_power,
            rho_gap_power=r.rho_gap_power,
            alternate_path_status=r.alternate_path_status,
            path_min_degree=r.path_min_degree,
            path_mean_degree=r.path_mean_degree,
            path_max_degree=r.path_max_degree,
            path_labels=r.path_labels,
            path_degrees=r.path_degrees,
        ))
    end

    return DataFrame(rows)
end

function summarize_by_distance_tier(pairwise::DataFrame)
    rows = NamedTuple[]
    isempty(pairwise) && return DataFrame(rows)

    for g in groupby(pairwise, [:object, :matrix, :tier, :endpoint_group, :distance])
        cov_values = [Float64(x) for x in skipmissing(g.cov) if isfinite(Float64(x))]
        corr_values = [Float64(x) for x in skipmissing(g.corr) if isfinite(Float64(x))]
        weighted_values = [Float64(x) for x in skipmissing(g.weighted_benchmark) if isfinite(Float64(x))]
        rho_values = [Float64(x) for x in skipmissing(g.rho_power) if isfinite(Float64(x))]
        isempty(cov_values) && continue
        isempty(corr_values) && continue

        push!(rows, (
            object=first(g.object),
            matrix=first(g.matrix),
            tier=first(g.tier),
            endpoint_group=first(g.endpoint_group),
            distance=first(g.distance),
            n_pairs=nrow(g),
            mean_cov=mean(cov_values),
            median_cov=median(cov_values),
            mean_abs_cov=mean(abs.(cov_values)),
            mean_corr=mean(corr_values),
            median_corr=median(corr_values),
            mean_abs_corr=mean(abs.(corr_values)),
            min_corr=minimum(corr_values),
            max_corr=maximum(corr_values),
            mean_rho_power=isempty(rho_values) ? NaN : mean(rho_values),
            mean_weighted_benchmark=isempty(weighted_values) ? NaN : mean(weighted_values),
        ))
    end

    out = DataFrame(rows)
    isempty(out) && return out
    sort!(out, [:object, :matrix, :tier, :endpoint_group, :distance])

    base_corr = Dict{Tuple{String,String,String,String},Tuple{Int,Float64}}()
    base_cov = Dict{Tuple{String,String,String,String},Tuple{Int,Float64}}()
    for g in groupby(out, [:object, :matrix, :tier, :endpoint_group])
        distances = collect(g.distance)
        base_distance = minimum(distances)
        base_row = g[g.distance .== base_distance, :][1, :]
        key = (String(base_row.object), String(base_row.matrix), String(base_row.tier), String(base_row.endpoint_group))
        base_corr[key] = (base_distance, Float64(base_row.mean_corr))
        base_cov[key] = (base_distance, Float64(base_row.mean_cov))
    end

    base_distance_col = Int[]
    mean_corr_ratio_to_base = Float64[]
    mean_cov_ratio_to_base = Float64[]
    for r in eachrow(out)
        key = (String(r.object), String(r.matrix), String(r.tier), String(r.endpoint_group))
        bd_corr, bcorr = base_corr[key]
        _, bcov = base_cov[key]
        push!(base_distance_col, bd_corr)
        push!(mean_corr_ratio_to_base, isfinite(bcorr) && bcorr != 0 ? r.mean_corr / bcorr : NaN)
        push!(mean_cov_ratio_to_base, isfinite(bcov) && bcov != 0 ? r.mean_cov / bcov : NaN)
    end
    out[!, :base_distance] = base_distance_col
    out[!, :mean_corr_ratio_to_base] = mean_corr_ratio_to_base
    out[!, :mean_cov_ratio_to_base] = mean_cov_ratio_to_base
    return out
end

function selected_paths_dataframe(records::Vector{<:NamedTuple})
    return DataFrame(records)
end

function empty_pairwise_frames()
    return DataFrame[], DataFrame[]
end

function write_outputs(out_dir::String,
                       meta,
                       selected_records::Vector{<:NamedTuple},
                       coverage_records::Vector{<:NamedTuple},
                       latent_indices::Vector{Int},
                       blocks::Dict{Symbol,Matrix{Float64}};
                       units::Symbol,
                       bins::Vector{DegreeBin},
                       options::Dict{String,Any})
    mkpath(out_dir)

    selected_paths = selected_paths_dataframe(selected_records)
    coverage = DataFrame(coverage_records)

    selected_paths_path = joinpath(out_dir, "selected_paths.csv")
    coverage_path = joinpath(out_dir, "coverage.csv")
    CSV.write(selected_paths_path, selected_paths)
    CSV.write(coverage_path, coverage)

    latent_parts = DataFrame[]
    outcome_parts = DataFrame[]
    for mode in sort!(collect(keys(blocks)); by=String)
        push!(latent_parts, latent_pairwise_dataframe(mode, blocks[mode], latent_indices, selected_records))
        push!(outcome_parts, outcome_pairwise_dataframe(mode, blocks[mode], latent_indices, selected_records))
    end

    latent_pairwise = isempty(latent_parts) ? DataFrame() : vcat(latent_parts...)
    outcome_pairwise = isempty(outcome_parts) ? DataFrame() : vcat(outcome_parts...)
    all_pairwise = if isempty(latent_pairwise)
        outcome_pairwise
    elseif isempty(outcome_pairwise)
        latent_pairwise
    else
        vcat(latent_pairwise, outcome_pairwise; cols=:union)
    end
    by_distance_tier = summarize_by_distance_tier(all_pairwise)

    latent_path = joinpath(out_dir, "latent_pairwise_decay.csv")
    outcome_path = joinpath(out_dir, "outcome_pairwise_decay.csv")
    by_distance_path = joinpath(out_dir, "by_distance_tier.csv")
    by_distance_cov_path = joinpath(out_dir, "by_distance_tier_cov.csv")
    by_distance_corr_path = joinpath(out_dir, "by_distance_tier_corr.csv")
    CSV.write(latent_path, latent_pairwise)
    CSV.write(outcome_path, outcome_pairwise)
    CSV.write(by_distance_path, by_distance_tier)
    CSV.write(by_distance_cov_path, Decay.covariance_summary_dataframe(by_distance_tier))
    CSV.write(by_distance_corr_path, Decay.correlation_summary_dataframe(by_distance_tier))

    output_files = Dict{String,Any}(
        "selected_paths_csv" => selected_paths_path,
        "coverage_csv" => coverage_path,
        "latent_pairwise_decay_csv" => latent_path,
        "outcome_pairwise_decay_csv" => outcome_path,
        "by_distance_tier_csv" => by_distance_path,
        "by_distance_tier_cov_csv" => by_distance_cov_path,
        "by_distance_tier_corr_csv" => by_distance_corr_path,
    )

    metadata = Dict(
        "estimates_path" => meta.estimates_path,
        "input_path" => meta.input_path,
        "outcome" => String(meta.outcome),
        "adjacency_weighting" => String(meta.a_weighting),
        "prior_adjacency" => String(meta.prior_adjacency),
        "obs_weighting" => String(meta.obs_weighting),
        "decomp_target" => String(meta.decomp_target),
        "rho_eps" => meta.rho_eps,
        "maxdeg" => meta.maxdeg,
        "rho" => meta.rho,
        "sigma_a" => meta.sigma_a,
        "sigma_z" => meta.sigma_z,
        "sigma_eps" => meta.sigma_eps,
        "y_std" => meta.y_std,
        "N_F" => meta.N_F,
        "N_M" => meta.N_M,
        "K" => meta.K,
        "n" => meta.n,
        "units" => String(units),
        "degree_bins" => Dict(bin_label(bin) => bin_spec(bin) for bin in bins),
        "bucket_scope" => get(options, "bucket_scope", "endpoint"),
        "anchor_selection" => get(options, "anchor_selection", "coverage"),
        "min_distances_per_series" => get(options, "min_distances_per_series", 2),
        "path_purity" => get(options, "bucket_scope", "endpoint") == "path" ?
            "Every node on the reconstructed shortest path must lie in the same requested degree bin." :
            "Only path endpoints must lie in the requested degree bin; internal path nodes may have any degree.",
        "second_shortest_mode" => get(options, "second_shortest", "off"),
        "isolation_eps" => get(options, "isolation_eps", nothing),
        "isolation_threshold_source" => get(options, "isolation_threshold_source", nothing),
        "isolation_threshold_base" => get(options, "isolation_threshold_base", nothing),
        "second_shortest_scope" => "T2 is computed with a native Yen K=2 implementation on the pre-selected paths in the full prior graph after sample/maxdeg filtering, not on the degree-tier subgraph.",
        "isolation_rule" => "Filter mode keeps a pre-selected path only when no second simple path exists or Yen's T2 is greater than max(T1, isolation_threshold_base).",
        "covariance_scope" => "Covariances are extracted from the full fitted GMRF precision; non-shortest walks are not removed.",
        "weighted_benchmark" => "rho^distance times the product of W edge weights along the selected shortest path, W=diag(df_is)*A*diag(dm_is).",
        "outcome_variance_scope" => "For outcome records, var_left, var_right, and corr use the latent component a_f + z_m only; sigma_eps residual variance is not added to the denominator.",
        "predictive_outcome_correlation_note" => "For raw predictive outcome correlations on distinct edges, residual covariance is zero but each variance includes residual noise, so correlations are lower in magnitude than the latent-component corr reported here.",
        "selected_path_count" => length(selected_records),
        "unique_latent_count" => length(latent_indices),
        "matrices_written" => sort!(String.(collect(keys(blocks)))),
        "options" => options,
        "output_files" => output_files,
    )

    metadata_path = joinpath(out_dir, "metadata.json")
    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)
    end
    output_files["metadata_json"] = metadata_path
    return output_files
end

function usage()
    println(
        "Usage:\n" *
        "  julia --project=. src/estimate/gmrf_degree_stratified_decay.jl [path/to/estimates.txt] [flags]\n" *
        "\n" *
        "Defaults target output/gmrfmle/full/mincomp-2-2/estimates.txt.\n" *
        "\n" *
        "Flags:\n" *
        "  --objects=latent|outcome|both             Objects to summarize (default: both)\n" *
        "  --matrix=prior|posterior|both             Covariance matrix to extract (default: prior)\n" *
        "  --sides=firms,persons                     Endpoint/anchor sides (default: firms,persons)\n" *
        "  --bins=low=1:2,medium=3:5,high=6:Inf      Degree bins (default shown)\n" *
        "  --bucket-scope=endpoint|path               Bucket paths by endpoints or require full path purity (default: endpoint)\n" *
        "  --anchor-selection=coverage|degree          Anchor ranking mode (default: coverage)\n" *
        "  --anchor-candidate-multiplier=N            Candidate multiplier for coverage anchor ranking (default: 25)\n" *
        "  --anchor-candidate-max=N                   Candidate cap for coverage anchor ranking (default: 250)\n" *
        "  --anchors-per-tier-side=N                 Anchors per tier and side (default: 5)\n" *
        "  --per-cell=N                              Selected paths per tier/side/distance cell (default: 25)\n" *
        "  --max-distance=N                          Maximum graph distance (default: 8)\n" *
        "  --min-distances-per-series=N               Drop series with fewer selected distances before extraction (default: 2)\n" *
        "  --second-shortest=filter|annotate|off      Second-shortest path isolation mode (default: filter)\n" *
        "  --isolation-eps=X                          Epsilon for automatic isolation threshold (default: 0.01)\n" *
        "  --isolation-threshold=auto|N               Base threshold K; auto uses ceil(log(eps)/log(abs(rho))) (default: auto)\n" *
        "  --units=original|scaled                    Output covariance units (default: original)\n" *
        "  --batch-size=N                             RHS batch size for solves (default: 8)\n" *
        "  --name=<label>                             Output label (default: degree_stratified_decay)\n" *
        "  --output-dir=<path>                        Override output directory\n" *
        "  --allow-sample-mismatch=true|false         Continue if sample metadata differs from estimates (default: false)\n"
    )
end

function main(args::Vector{String}=ARGS)
    if any(arg -> arg in ("--help", "-h"), args)
        usage()
        return 0
    end

    parsed = parse_flags(args)
    estimates_path = CovTool.repo_path(parsed.estimates_path)
    flags = parsed.flags
    isfile(estimates_path) || error("Estimates file not found: $(estimates_path)")

    objects = parse_objects(get(flags, "--objects", "both"))
    matrix_mode = Symbol(get(flags, "--matrix", "prior"))
    Decay.matrix_modes(matrix_mode)

    sides = Decay.parse_sides(get(flags, "--sides", "firms,persons"))
    bins = haskey(flags, "--bins") ? parse_degree_bins(flags["--bins"]) : default_degree_bins()

    bucket_scope = parse_bucket_scope(get(flags, "--bucket-scope", "endpoint"))
    anchor_selection = parse_anchor_selection(get(flags, "--anchor-selection", "coverage"))
    anchor_candidate_multiplier = Decay.parse_int_flag(flags, "--anchor-candidate-multiplier", 25; min_value=1)
    anchor_candidate_max = Decay.parse_int_flag(flags, "--anchor-candidate-max", 250; min_value=1)
    anchors_per_tier_side = Decay.parse_int_flag(flags, "--anchors-per-tier-side", 5; min_value=1)
    per_cell = Decay.parse_int_flag(flags, "--per-cell", 25; min_value=1)
    max_distance = Decay.parse_int_flag(flags, "--max-distance", 8; min_value=1)
    min_distances_per_series = Decay.parse_int_flag(flags, "--min-distances-per-series", 2; min_value=1)
    batch_size = Decay.parse_int_flag(flags, "--batch-size", 8; min_value=1)
    allow_sample_mismatch = Decay.parse_bool_flag(flags, "--allow-sample-mismatch", false)
    units = Symbol(get(flags, "--units", "original"))
    units in (:original, :scaled) || error("--units must be original or scaled; got $(units)")
    second_shortest_mode = parse_second_shortest_mode(get(flags, "--second-shortest", "filter"))
    isolation_eps = parse_isolation_eps(get(flags, "--isolation-eps", "0.01"))
    isolation_threshold_raw = get(flags, "--isolation-threshold", "auto")

    name = get(flags, "--name", "degree_stratified_decay")
    out_dir = haskey(flags, "--output-dir") ?
        CovTool.repo_path(flags["--output-dir"]) :
        CovTool.default_output_dir(estimates_path, name)

    @printf("Reading estimates: %s\n", estimates_path)
    meta = CovTool.parse_estimates_file(estimates_path)
    threshold_source, threshold_base = if second_shortest_mode == :off
        ("off", nothing)
    else
        parse_isolation_threshold(isolation_threshold_raw, meta.rho, isolation_eps)
    end
    second_settings = second_shortest_settings(
        second_shortest_mode,
        isolation_eps,
        threshold_source,
        threshold_base,
    )

    input_path = CovTool.repo_path(meta.input_path)
    isfile(input_path) || error("Input parquet not found: $(input_path)")

    @printf("Reading sample parquet: %s\n", input_path)
    df = Parquet2.readfile(input_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    if meta.maxdeg !== nothing
        df = CovTool.filter_maxdeg(df, meta.maxdeg; verbose=true)
        nrow(df) > 0 || error("No rows remain after maxdeg=$(meta.maxdeg)")
    end

    prepared = CovTool.prepare_data_with_ids(
        df;
        outcome=meta.outcome,
        a_weighting=meta.a_weighting,
        prior_adjacency=meta.prior_adjacency,
        obs_weighting=meta.obs_weighting,
        rho_eps=meta.rho_eps,
        verbose=true,
    )
    prep = prepared.prep
    CovTool.validate_reconstructed_sample(meta, prep;
                                          strict=!allow_sample_mismatch,
                                          firms=prepared.firms,
                                          people=prepared.people)

    neighbors = Decay.bipartite_neighbors(prep)
    selected_records = NamedTuple[]
    coverage_records = NamedTuple[]

    if :latent in objects
        @printf("Selecting %s-bucket latent paths...\n", String(bucket_scope))
        records, coverage = select_latent_paths(
            prepared,
            neighbors,
            bins;
            sides=sides,
            anchors_per_tier_side=anchors_per_tier_side,
            per_cell=per_cell,
            max_distance=max_distance,
            rho=meta.rho,
            second_shortest_settings=second_settings,
            bucket_scope=bucket_scope,
            anchor_selection=anchor_selection,
            anchor_candidate_multiplier=anchor_candidate_multiplier,
            anchor_candidate_max=anchor_candidate_max,
        )
        append!(selected_records, records)
        append!(coverage_records, coverage)
    end

    if :outcome in objects
        @printf("Selecting %s-bucket outcome paths...\n", String(bucket_scope))
        records, coverage = select_outcome_paths(
            prepared,
            neighbors,
            bins;
            endpoint_sides=sides,
            anchors_per_tier_side=anchors_per_tier_side,
            per_cell=per_cell,
            max_distance=max_distance,
            rho=meta.rho,
            second_shortest_settings=second_settings,
            bucket_scope=bucket_scope,
            anchor_selection=anchor_selection,
            anchor_candidate_multiplier=anchor_candidate_multiplier,
            anchor_candidate_max=anchor_candidate_max,
        )
        append!(selected_records, records)
        append!(coverage_records, coverage)
    end

    selected_records, coverage_records = filter_records_for_decay(
        selected_records,
        coverage_records;
        min_distances_per_series=min_distances_per_series,
    )

    isempty(selected_records) && error("No paths remain after distance-support filtering; inspect coverage or lower --min-distances-per-series.")
    latent_indices = latent_indices_for_records(selected_records)

    @printf("Selected paths: %d\n", length(selected_records))
    @printf("Unique latent nodes required for covariance extraction: %d\n", length(latent_indices))
    @printf("Output directory: %s\n", out_dir)

    blocks = extract_sigma_blocks(
        prep,
        meta,
        latent_indices;
        matrix_mode=matrix_mode,
        units=units,
        batch_size=batch_size,
    )

    options = Dict{String,Any}(
        "objects" => String.(objects),
        "matrix" => String(matrix_mode),
        "sides" => String.(sides),
        "bins" => Dict(bin_label(bin) => bin_spec(bin) for bin in bins),
        "bucket_scope" => String(bucket_scope),
        "anchor_selection" => String(anchor_selection),
        "anchor_candidate_multiplier" => anchor_candidate_multiplier,
        "anchor_candidate_max" => anchor_candidate_max,
        "anchors_per_tier_side" => anchors_per_tier_side,
        "per_cell" => per_cell,
        "max_distance" => max_distance,
        "min_distances_per_series" => min_distances_per_series,
        "second_shortest" => String(second_settings.mode),
        "isolation_eps" => isolation_eps,
        "isolation_threshold" => isolation_threshold_raw,
        "isolation_threshold_source" => second_settings.threshold_source,
        "isolation_threshold_base" => second_settings.threshold_base,
        "batch_size" => batch_size,
        "name" => name,
        "allow_sample_mismatch" => allow_sample_mismatch,
    )
    written = write_outputs(
        out_dir,
        meta,
        selected_records,
        coverage_records,
        latent_indices,
        blocks;
        units=units,
        bins=bins,
        options=options,
    )

    @printf("Wrote selected paths: %s\n", written["selected_paths_csv"])
    @printf("Wrote coverage: %s\n", written["coverage_csv"])
    @printf("Wrote latent pairwise decay: %s\n", written["latent_pairwise_decay_csv"])
    @printf("Wrote outcome pairwise decay: %s\n", written["outcome_pairwise_decay_csv"])
    @printf("Wrote by-distance/tier summary: %s\n", written["by_distance_tier_csv"])
    @printf("Wrote metadata: %s\n", written["metadata_json"])
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(GMRFDegreeStratifiedDecay.main())
end
