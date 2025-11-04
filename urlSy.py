# Create a text file with ready-to-use Symbolab URLs (LaTeX already URL-encoded)
from urllib.parse import quote

latex_z = "z = -0.140762 + (0.212931)\\,\\mathrm{male} + (0.507331)\\,\\mathrm{age} + (0.024948)\\,\\mathrm{currentSmoker} + (0.225761)\\,\\mathrm{cigsPerDay} + (0.030697)\\,\\mathrm{BPMeds} + (0.032709)\\,\\mathrm{prevalentStroke} + (0.119374)\\,\\mathrm{prevalentHyp} + (0.032995)\\,\\mathrm{diabetes} + (0.149440)\\,\\mathrm{totChol} + (0.306061)\\,\\mathrm{sysBP} + (0.003341)\\,\\mathrm{diaBP} + (0.058479)\\,\\mathrm{BMI} + (-0.049004)\\,\\mathrm{heartRate} + (0.104115)\\,\\mathrm{glucose} + (-0.225629)\\,\\mathrm{education}_{2.0} + (-0.285119)\\,\\mathrm{education}_{3.0} + (-0.153907)\\,\\mathrm{education}_{4.0}"
latex_p = "\\Pr(\\mathrm{TenYearCHD}=1) = \\frac{1}{1 + e^{-z}}"

base = "https://es.symbolab.com/solver/step-by-step/"
url_z = base + quote(latex_z, safe='') + "?or=input"
url_p = base + quote(latex_p, safe='') + "?or=input"

content = f"""URLs listas para pegar en el navegador (Symbolab):

1) Ecuación del logit (z):
{url_z}

2) Probabilidad sigmoide:
{url_p}

También puedes combinar ambas expresiones con un salto de línea codificado (%0A), por ejemplo:
{base}{quote(latex_z + " %0A " + latex_p, safe='')}?or=input
"""

path = "/home/xander/url.txt"

with open(path, "w") as f:
    f.write(content)

path
