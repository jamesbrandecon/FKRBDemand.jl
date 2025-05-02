using Documenter
using FKRBDemand

makedocs(
    sitename = "FKRBDemand.jl",
    format = Documenter.HTML(),
    modules = [FKRBDemand], 
    warnonly = true,
    pages = [
        "Home" => "index.md"]
)

deploydocs(
    repo = "github.com/jamesbrandecon/FKRBDemand.jl.git",
    push_preview = true,
    devbranch = "main"
)