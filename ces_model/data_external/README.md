# External data (not included)

`sim10_diederich_estimation.py` and `sim10b_diederich_profile.py` read
`data_external/diederich_etal_dataverse/data/WS_data_noIP_noMTID_panel.dta`.

**Article.** Diederich, J., Eckel, C. C., Epperson, R., Goeschl, T. and Grossman, P. J. (2022). Subsidizing unit donations:
matches, rebates, and discounts compared. *Experimental Economics* 25, 734-758. https://doi.org/10.1007/s10683-021-09732-9

**Data.** Diederich, J., Eckel, C. C., Epperson, R., Goeschl, T. and Grossman, P. J. (2021). Subsidizing Unit Donations:
Matches, Rebates, and Discounts Compared. Replication data, version 1.1. heiDATA, Heidelberg Research Data Repository.
https://doi.org/10.11588/DATA/GANTNY (access restricted; raw data are not redistributed here).

File used: `WS_data_noIP_noMTID_panel.dta` (within-subject sample; 113 participants, 7 environments, 791 observations,
one online session on 9 July 2015). Precomputed outputs are in `../data/diederich_*.csv`.

## Further data used
- Between-subjects file `BS_data_noIP_noMTID.dta` of the same Diederich et al. package (558 participants) for `sim11_diederich_between.py`.
- Karlan, D. and List, J. A. (2007), Does price matter in charitable giving? Evidence from a large-scale natural field experiment, *American Economic Review* 97(5), 1774-1793. Data: Karlan, D. and List, J. A. (2014), Harvard Dataverse, V4, https://doi.org/10.7910/DVN/27853 (UNF:5:C63Hp69Cn88CkFCoOA0N/w==; Innovations for Poverty Action Dataverse). File `AER merged.dta`, place in `data_external/karlan_list_2007_dataverse/` for `sim12_karlan_list.py`.

## Additional open data collected (not yet used in any script)
Raw files are not stored in this repository. Citations to be completed before any use.
- Zhu, Z. (2026). Replication package: Decision time is associated with future cooperation and social rewiring decisions in network public goods games. Zenodo, v1, CC BY 4.0. https://doi.org/10.5281/zenodo.19544006 (supplement to PLOS ONE, doi:10.1371/journal.pone.0347919). Binary cooperation, 1,462 subjects, 10 rounds.
- Ottoni-Wilhelm, M., Vesterlund, L. and Xie, H. (2017). Why do people give? Testing pure and impure altruism. *American Economic Review* 107(11), 3617-3633. Replication data: openICPSR project 113005, V1 (files `voxA`; DOI to be confirmed). 85 subjects, 6 budgets.
- Ando, H., Nishi, A. and Handcock, M. S. (2025). Statistical modeling of networked evolutionary public goods games. arXiv:2501.07007; code and data: github.com/Ankoudon/ando2025ergm (release v1.0.0). 20 dynamic networks.
- OSF archives (15 public goods game packages, identified only by OSF code, e.g. `9br85`: cost-(in)effective public good provision, 308 subjects, four return levels): see `VERI_ENVANTERI_27Ekim.md` in the local `data_external` folder. Bibliographic details to be verified.
