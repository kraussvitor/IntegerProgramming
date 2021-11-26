#=
- **User cuts**: Ajuda a fortalecer um MIP removendo soluções fracionárias. Os cortes **não** são necessários para o modelo, mas podem ajudar um MIP a resolver mais rápido.

- MOI.set(model, MOI.RawParameter("Precrush"), 1)

- **Lazy constraints**: São necessárias para o modelo: o modelo estaria incorreto sem essas restrições. Normalmente são utilizadas em modelos que contém um número relativamente grande de restrições que não são facilmente satisfeitas. 

- MOI.set(model, MOI.RawParameter("LazyConstraints"), 1)

Parâmetros: https://www.gurobi.com/documentation/9.1/refman/parameters.html
=#

#

using Gurobi, JuMP
using BenchmarkTools
include("cats_reader.jl")
include("maximum_weight_clique.jl")

function branch_and_cut(w, A)
    n, m = size(A)
    ConflictGraph = weighted_conflict_graph(A)
    # declaring the model
    model = Model(Gurobi.Optimizer)
    #set_optimizer_attribute(model, "PreCrush", 1) #Para usar UserCuts
    #set_optimizer_attribute(m, "Cuts", 0) #Desabilitar outros cortes
    #set_optimizer_attribute(m, "Presolve", 0) #Desabilitar presolve
    #set_optimizer_attribute(m, "Heuristics", 0) #Desabilitar heurística
    ## Definition of solver parameters for own branch-and-cut
    #MOI.set(model, MOI.RawParameter("PreCrush"), 1) # Habilitar cortes do tipo UserCuts
    MOI.set(model, MOI.RawParameter("Cuts"), 1) # Desabilitar cortes
    MOI.set(model, MOI.RawParameter("Presolve"), 1) # Desabilitar presolve
    MOI.set(model, MOI.RawParameter("Heuristics"), 1) # Desabilitar heurísticas
    #MOI.set(model, MOI.RawParameter("OutputFlag"), 0) # Desabilitar log
    #MOI.set(model, MOI.RawParameter("LogFile"), "BC_0003_All_2.txt") # Save log to file
    #MOI.set(model, MOI.RawParameter("LogToConsole"), 0) #Log not to the console
    MOI.set(model, MOI.RawParameter("TimeLimit"), 100) # Time limit
    #set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer, "LogFile" => "BC_0000_false", "LogToConsole" => 0, "TimeLimit" => 10))
    ## Definition of the model
    @variable(model, x[i in 1:n], Bin)
    @constraint(model, cover[j in 1:m], sum(A[i,j] * x[i] for i in 1:n) <= 1)
    @objective(model, Max, sum(w[i] * x[i] for i in 1:n))
    ## callback function
    function my_callback_function(cb_data)
        x_vals = callback_value.(Ref(cb_data), x)
        update_weights(ConflictGraph, x_vals)
        max_clique = maximum_weight_clique(ConflictGraph; store=false)
        clique_weight = subset_weight(ConflictGraph, max_clique)
        if clique_weight > 1
            cut = @build_constraint(
                sum(x[i] for i in max_clique) <= 1 
            )
            MOI.submit(model, MOI.UserCut(cb_data), cut)
        end
    end

    ## Defition of the user cut function
    #MOI.set(model, MOI.UserCutCallback(), my_callback_function)
    #MOI.set(model, Gurobi.CallbackFunction(), my_callback_function)
    # solve
    optimize!(model)
    println("Gurobi time = ", solve_time(model))
    println("Gurobi z = ", dot(w, value.(x)))
end

folder = "CAArbitraryBig\\"
file = "arbitrary_dist_big_0000.txt"
n, m, w, A = read_cats_instance(folder * file)
branch_and_cut(w, A)