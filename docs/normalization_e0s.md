# Overview of QM Methods and Normalization

OpenQDC provides support for 250+ QM Methods and provides a way to standardize and categorize
the usage of different level of theories used for Quantum Mechanics Single Point Calculations
to add value and information to the datasets.

## Level of Theory

To avoid inconsistencies, level of theories are standardized and categorized into Python Enums
consisting of a functional, a basis set, and a correction method.
OpenQDC covers more than 106 functionals, 20 basis sets, and 11
correction methods.
OpenQDC provides the computed the isolated atom energies `e0` for each QM method.


## Normalization


We provide support of energies through "physical" and "regression" normalization to conserve the size extensivity of chemical systems.
OpenQDC through this normalization, provide a way to transform the potential energy to atomization energy by subtracting isolated atom energies `e0`
physically interpretable and extensivity-conserving normalization method. Alternatively, we pre-
compute the average contribution of each atom species to potential energy via linear or ridge
regression, centering the distribution at 0 and providing uncertainty estimation for the computed
values. Predicted atomic energies can also be scaled to approximate a standard normal distribution.

### Physical Normalization

`e0` energies are calculated for each atom in the dataset at the appropriate level of theory and then subtracted from
the potential energy to obtain the atomization energy. This normalization method is physically interpretable and
only remove the atom energy contribution from the potential energy.


### Regression Normalization

`e0` energies are calculated for each atom in the dataset from fitting a regression model to the potential energy.
The `e0` energies are then subtracted from the potential energy to obtain the atomization energy. This normalization
provides uncertainty estimation for the computed values and remove part of the interatomic energy contribution from the potential energy.
The resulting formation energy is centered at 0.
