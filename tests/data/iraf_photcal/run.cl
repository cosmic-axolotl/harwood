noao
digiphot
photcal
fitparams ("observations.obsfile", "catalogo.mkcatalog", "configuration.mkconfig", "fit1.txt", weighting="uniform", interactive=no, logfile="fit1.log", log_fit=yes, log_results=yes)
invertfit ("observations.obsfile", "configuration.mkconfig", "fit1.txt", "inv1.txt", catalogs="catalogo.mkcatalog")
logout
