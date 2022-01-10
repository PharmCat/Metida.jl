struct FormulaException <: Exception
    msg::String
end
Base.showerror(io::IO, e::FormulaException) =
    print(io, "FormulaError: ", e.msg)
