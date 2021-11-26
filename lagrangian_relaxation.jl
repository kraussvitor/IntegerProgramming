using LinearAlgebra
using Distributions
using Random
using JuMP, Gurobi
using DataStructures
using SimpleGraphs
using LightGraphs
using DisjointCliqueCover
using GraphPlot
using BenchmarkTools
include("maximum_weight_clique.jl")

function solve_relaxed_problem(u, v, w, A, B)
    n, m = size(A)
    sol = zeros(n)
    for i = 1:n
        s = w[i] - dot(u, A[i,:]) - dot(v, B[:,i])
        if s > 0
            sol[i] = 1
        end
    end    
    return sol
end

function u_zero(w, A)
    n, m = size(A)
    u = zeros(m)
    for j = 1:m
        s = 0
        for i = 1:n
            if A[i,j] > 0
                s += w[i] / sum(A[i,:])
            end
        end
        u[j] = s / maximum([sum(A[:,j]), 1]) 
    end
    return u
end

function subgradient_u(x, A)
    n, m = size(A)
    return ones(m) - A' * x
end

function subgradient_v(x, B)
    k, l = size(B)
    return ones(k) - B * x   
end

function greater_than(vec, bound)
    n = size(vec, 1)
    b = zeros(n)
    for i = 1:n
        if vec[i] > bound
            b[i] = 1
        end
    end
    return b
end

function find_feasible_solution(x, w, A)
    n, m = size(A)
    W = maximum(w)
    new_x = copy(x)
    times_covered = A' * x
    T = greater_than(times_covered, 1)
    active_indices = [i for i in 1:n if x[i] > 0]
    while sum(T) > 0
        index_min = 1
        value_min = W + 1
        for i in active_indices
            if new_x[i] > 0
                c = w[i] / (dot(T, A[i,:]))
                if c < value_min
                    index_min = i
                    value_min = c
                end
            end
        end
        new_x[index_min] = 0
        times_covered -= A[index_min,:]
        T = greater_than(times_covered, 1)
    end
    return new_x
end

function edge_list(A)
    n, m = size(A)
    edge_list = []
    for i = i:n
        for j = 1:n
            if dot(A[i,:], A[j,:]) > 0 && i < j
                push!(edge_list, (i,j))
            end
        end
    end
    return edge_list
end

function conflict_graph(A)
    n, m = size(A)
    g = LightGraphs.Graph(n)
    edges = edge_list(A)
    for e in edges
        add_edge!(g, e[1], e[2])
    end
    return g
end

function clique_cover_matrix(A)
    n, m = size(A)
    g = weighted_conflict_graph(A)
    max_clique , cliques = maximum_weight_clique(g; store=true)
    B = Array{Bool}(undef, 0, n)
    for c in cliques
        if length(c) > 2
            row = zeros(n)
            for j in c
                row[j] = 1
            end
            B = [B; row']
        end
    end
    return B
end

function obj_value(x, w)
    return dot(x, w)
end

function lagrangian_value(u, v, x, w, A, B)
    n, m = size(A)
    k, l = size(B)
    s = sum(u) + sum(v)
    for i = 1:n
        if x[i] > 0
            for j = 1:m
                s -= A[i,j] * u[j]
            end
            for j = 1:k
                s -= B[j,i] * v[j]
            end
            s += w[i]
        end
    end
    return s
end

function update_u(u, step_size, g)
    m = size(u, 1)
    for i = 1:m
        u[i] = maximum([0, u[i] - step_size * g[i] ])  
    end
    return u
end

function update_v(v, step_size, g)
    k = size(v, 1)
    for i = 1:k
        v[i] = maximum([0, v[i] - step_size * g[i] ])
    end
    return v
end

function appended_norm(u, v)
    s = 0.0
    for i in u
        s += i ^ 2
    end
    for i in v
        s += i ^ 2
    end
    if s > 0
        return sqrt(s)
    end
    return 1.0  
end

function laha(w, A)
    n, m = size(A)
    B = clique_cover_matrix(A)
    k, l = size(B)
    u = u_zero(w, A)
    v = zeros(k)
    F = 0.25
    t = 0
    best_val = -1
    best_sol = nothing
    best_upper = sum(w)
    best_lower = 0
    epsilon = 0.9
    while F > 0.05  && best_upper - best_lower > epsilon
        relaxed_x = solve_relaxed_problem(u, v, w, A, B)
        feasible_x =  find_feasible_solution(relaxed_x, w, A)
        l = lagrangian_value(u, v, relaxed_x, w, A, B)
        step_size = F * (l - best_lower) / appended_norm(u, v)
        upper = l #minimum([l, lagrangian_value(u, v, feasible_x, w, A, B)])
        lower = obj_value(feasible_x, w)
        g_u = subgradient_u(relaxed_x, A)
        g_v = subgradient_v(relaxed_x, B)
        u = update_u(u, step_size, g_u)
        v = update_v(v, step_size, g_v)
        if upper < best_upper
            best_upper = upper
        end
        if lower > best_lower
            best_lower = lower
        end
        if lower > best_val
            best_val = lower
            best_sol = feasible_x
            t = 0
        else
            t += 1
        end
        if t > 100
            F /= 2.0
            t = 0
        end
    end
    return best_sol, best_upper, best_lower
end

function swap(x, in, out)
    new_x = copy(x)
    new_x[in] .= 1
    new_x[out] .= 0
    return new_x
end

function is_feasible(x, A)
    times_covered = A' * x
    return maximum(times_covered) < 2
end

function avg_price(x, w, A)
    bids_covered = sum(A' * x)
    return obj_value(x, w) / maximum([1, bids_covered])
end

function better_sol(x, new_x, w, A)
    z1 = obj_value(x, w)
    z2 = obj_value(new_x, w)
    a1 = avg_price(x, w, A)
    a2 = avg_price(new_x, w, A)
    return (z1 < z2) || (a1 < a2)
end

function push_neighbour(stack, x, in, out, w, A)
    new_x = swap(x, in, out)
    if is_feasible(new_x, A) && better_sol(new_x, x, w, A)
        push!(stack, new_x)
    end
end

function local_search(x, w, A)
    n, m = size(A)
    best_sol = x
    best_value = obj_value(x, w)
    t = 0
    d = 0
    s = Stack{Vector}()
    push!(s, x)
    max_iter = 10 # ceil(0.1 * n)
    while !(isempty(s)) && t < max_iter
        current = pop!(s)
        z = obj_value(current, w)
        if z > best_value
            best_sol = current
            best_value = z
            t = 0
        else
            t += 1
        end
        for i = 1:n
            if current[i] > 0
                for j = 1:n
                    for k = 1:j
                        b1 = (j != i && current[j] == 0)
                        b2 = (k != i && current[k] == 0)
                        b3 = j != k
                        if  b1 && b2 && b3
                            out = [i]
                            in = [j,k]
                            push_neighbour(s, current, in, out, w, A)
                        end
                    end
                end
            end
        end
    end
    return best_sol
end

function test_generator(n, m, p)
    b = Bernoulli(p)
    u = Uniform(0.9, 1.1)
    A = rand(b, (n,m))
    job_prices = rand(u, (m)) .* (A' * ones(n))
    bid_prices = rand(u, (n)) .* (A * job_prices)
    return bid_prices, A
end

function gurobi_solve(w, A; timelimit=100)
    n, m = size(A)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attributes(model) # "OutputFlag" => 0
    set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => timelimit, "OutputFlag" => 0)) # "OutputFlag" => 0
    @variable(model, x[i in 1:n], Bin)
    @constraint(model, cover[j in 1:m], sum(A[i,j] * x[i] for i in 1:n) <= 1)
    @objective(model, Max, sum(w[i] * x[i] for i in 1:n))
    optimize!(model)
    return value.(x), objective_value(model), solve_time(model)
end