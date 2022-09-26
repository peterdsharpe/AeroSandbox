### A few functions to take a WAG at the *technological* feasibility of
#   a nuclear-RTG-powered airplane. Not at all endorsing this as a good idea,
#   just a fascinating one that's surprisingly technologically feasible with Polonium-210.

# Aircraft uses:
# There's speculation that the Russians might already be flying this?
# Source (Overview): https://theconversation.com/nuclear-powered-missile-accident-in-russia-what-really-happened-121966
# Source (Russian press release): http://rosatom.ru/journalist/news/zayavlenie-departamenta-kommunikatsiy-goskorporatsii-rosatom/
# Source (American perspective): https://foreignpolicy.com/2019/08/12/russia-mysterious-explosion-arctic-putin-chernobyl/

# How it's made:
# Po-210 is created by neutron bombardment of (stable) bismuth-209 in a nuclear reactor.
# This forms radioactive bismuth-210 (half-life: 5 days), which decays to Po-210 via beta decay.
# Source: https://personal.ems.psu.edu/~radovic/Polonium_specs.pdf

# Cost:
# Po-210 appears to be affordable, at least in milligram amounts (used in manufacturing
# processes for things as simple as static elimination).
# Source: https://personal.ems.psu.edu/~radovic/Polonium_specs.pdf

# Safety:
# Alpha emitter, so safe as long as it's not ingested or inhaled. If ingested, it has a LD50 of 0.7 micrograms.
# If inhaled, it's about 20% of that.
# Source: https://personal.ems.psu.edu/~radovic/Polonium_specs.pdf

# Supply:
# Only 100 grams of Po-210 is produced each year, almost all in Russia.
# Source: https://www.nrc.gov/reading-rm/doc-collections/fact-sheets/polonium.html
# This seems to be an insurmountable issue.

alpha_particle_mass_amu = 4.001506179127  # mass of an alpha particle
po_210_mass_amu = 209.9828736  # mass of Polonium-210
pb_206_mass_amu = 205.9744818  # mass of Lead-206
amu = 1.6605390666050e-27  # kg, 1 atomic mass unit
c = 299792458  # m/s, speed of light


def po210_specific_power(
        days_after_formation=0,
):
    half_life = 138.376  # days
    # Source: https://en.wikipedia.org/wiki/Polonium-210
    pure_specific_energy = (
            (po_210_mass_amu - alpha_particle_mass_amu - pb_206_mass_amu) * c ** 2
            / po_210_mass_amu
    )  # J/kg
    # TODO finish
