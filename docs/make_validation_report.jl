################################################################################
#                      СБОРКА ВАЛИДАЦИОННОГО ОТЧЕТА
################################################################################

using Metida, Weave

const RDIR = joinpath(dirname(pathof(Metida)), "..", "test")
const DOUT = normpath(joinpath(RDIR, "..", "docs", "src"))

include(joinpath(RDIR, "acceptance.jl"))

doctype = length(ARGS) >= 1 ? ARGS[1] : "md"
kind = doctype == "pdf"  ? "md2pdf" :
       doctype == "html" ? "md2html" :
       doctype == "md"   ? "github" :
       error("неизвестный формат: $doctype (ожидается md, pdf или html)")

isdir(DOUT) || mkpath(DOUT)

println("сборка: ", joinpath(RDIR, "validation_report.jmd"), " → ", DOUT, " [", kind, "]")

weave(joinpath(RDIR, "validation_report.jmd");
      doctype   = kind,
      out_path  = DOUT,
      pandoc_options = doctype == "pdf" ? ["--toc", "-V colorlinks=true"] : String[])

ok, txt = acceptance_verdict_from_files()
println("\n", "="^78, "\n", txt, "\n", "="^78)

strict = get(ENV, "VALIDATION_STRICT", "true") != "false"
if !ok && strict
    error("валидационный отчет не принят; VALIDATION_STRICT=false оставит его для разбора")
end
