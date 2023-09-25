from math import pi

# Lengths (equivalents in meters)
meter = 1
inch = 0.0254
foot = 12 * inch
yard = 3 * foot
mile = 5280 * foot
naut_mile = 1852

# Masses (equivalents in kilograms)
kg = 1
gram = 0.001
slug = 14.59390
lbm = 0.45359237
pound = lbm
short_ton = 2000 * lbm
long_ton = 2240 * lbm
oz = lbm / 16

# Time (equivalents in seconds)
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365.2425 * day
month = year / 12

# Force (equivalents in Newtons)
newton = 1
lbf = slug * foot
pound_force = lbf

# Speed (equivalents in m/s)
kph = 1000 * meter / hour
knot = 1.852 * kph
mph = mile / hour
fps = foot / second

# Rotational Speed (equivalents in radians/second)
rads_per_sec = 1
rps = 2 * pi
rpm = rps / 60

# Volume (equivalents in m^3)
liter = 0.001
gallon_us = 231 * inch ** 3
gallon_imperial = 4.54609 * liter
gallon = gallon_us
quart = gallon_us / 4

# Pressure (equivalents in Pa)
pascal = 1
atm = 101325
torr = atm / 760
psi = lbf / inch ** 2
psf = lbf / foot ** 2

# Power (equivalents in Watts)
watt = 1
horsepower = 550 * foot * lbf / second
hp = horsepower

# Current (equivalents in Amperes)
amp = 1

# Energy (equivalents in Joules)
joule = 1
btu = 1055.05585262
calorie = 4.184
kcal = 1000 * calorie
watt_hour = watt * hour

# SI prefixes
yocto = 1e-24
zepto = 1e-21
atto = 1e-18
femto = 1e-15
pico = 1e-12
nano = 1e-9
micro = 1e-6
milli = 1e-3
centi = 1e-2
deci = 1e-1
deka = 1e1
hecto = 1e2
kilo = 1e3
mega = 1e6
giga = 1e9
tera = 1e12
peta = 1e15
exa = 1e18
zetta = 1e21
yotta = 1e24

# Some abbreviations / alternate names
m = meter
ft = foot
yd = yard
mi = mile
nmi = naut_mile
kilogram = kg
sec = second
s = second
min = minute
hr = hour
wk = week
yr = year
N = newton
kt = knot
L = liter
Pa = pascal
