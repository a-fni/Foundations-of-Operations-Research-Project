# Foundations of Operations Research Project

This repository contains the source code the final project for the [Foundations of Operations Research](https://onlineservices.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2025&k_cf=225&k_corso_la=542&k_indir=T2I&codDescr=088983&lang=IT&semestre=1&anno_corso=1&idItemOfferta=177910&idRig)
course at Politecnico di Milano.

## Overview

The [Foundations of Operations Research](https://onlineservices.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2025&k_cf=225&k_corso_la=542&k_indir=T2I&codDescr=088983&lang=IT&semestre=1&anno_corso=1&idItemOfferta=177910&idRig)
course includes a hands-on lab where optimization problems are solved in an automated way using python. Models are described using the [`mip`](https://www.python-mip.com/)
library and a final evaluation of the lab, worth 4 points, is carried out during the exam session. As an alternative the final exam session evaluations students may develop a small project
instead. For the academic year 2025-2026 the small project assigned is described in the Small_Project_Description.pdf file and, in summary, requires students to solve an mTSP problem.

More in detail, a set of `k = 4` drones must navigate a provided graph so as to visit collaboratively all nodes of the graph. The trip of each drone must start and end at a provided intial depot node.
The optimization process should attempt to minimize the overall max-span travel time, namely the maximum travel time across all drone travel times.

The problem at hand is a complex NP-Hard problem that should be tackled either with heuristic algorithms (more efficient but potentially leading to sub-optimal solutions) or with mixed-integer linear programming
algorithms. The former option was adopted by the solution provided in the current repository.

In order to evaluate the model during development two open-test cases were provided, respectively contained in the `Edificio1.csv` and `Edificio2.csv` files.

## Building

To build and run the optimization script we can simply create a python virtual environment and install the required dependencies listed inside the `requirements.txt` file:

`python3 -m venv venv && source venv/bin/activate && pip3 install -r requirements.txt`

Note that the [`mip`](https://www.python-mip.com/) python library does have some compatibility issues on certain hardware (for example, on Apple Silicon machines).

## Running the script

As described in the Small_Project_Description.pdf file the script is run by means of the following syntax:

`python3 main.py <FILENAME.csv>`

Where `FILENAME.csv` is the name of the `csv` dataframe that the script must read in order to import coordinate values and should be a value among `Edificio1.csv` and `Edificio2.csv`.

## Deliverable submitted

As described in the Small_Project_Description.pdf to submit the final optimization script it is required to create a zip archive named according to the format `SURNAME_NAME_STUDENTID.zip` containing:

```
SURNAME_NAME_STUDENTID/
├─ Edificio1.csv
├─ Edificio2.csv
├─ main.py
└─ participants.txt
```

Where the two `csv` files contain the open test cases and `participants.txt` file contains the list of students (maximum of 4) who worked on the project.
