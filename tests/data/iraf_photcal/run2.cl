noao
digiphot
photcal
fitparams ("observations.obsfile", "catalogo.mkcatalog", "configuration.mkconfig", "fitB.txt", weighting="photometric", addscatter=yes, interactive=no, logfile="fitB.log", log_fit=yes, log_results=yes)
fitparams ("observations.obsfile", "catalogo.mkcatalog", "configuration.mkconfig", "fitC.txt", weighting="photometric", addscatter=no, interactive=no, logfile="fitC.log", log_fit=yes, log_results=yes)
fitparams ("observations.obsfile", "catalogo.mkcatalog", "configuration.mkconfig", "fitD.txt", weighting="uniform", nreject=3, low_reject=1.5, high_reject=1.5, interactive=no, logfile="fitD.log", log_fit=yes, log_results=yes)
fitparams ("observations.obsfile", "catalogo.mkcatalog", "config2.cfg", "fitE.txt", weighting="equations", interactive=no, logfile="fitE.log", log_fit=yes, log_results=yes)
fitparams ("observations.obsfile", "catalogo.mkcatalog", "config2.cfg", "fitF.txt", weighting="photometric", interactive=no, logfile="fitF.log", log_fit=yes, log_results=yes)
fitparams ("obs2.txt", "catalogo.mkcatalog", "configuration.mkconfig", "fitG.txt", weighting="uniform", interactive=no, logfile="fitG.log", log_fit=yes, log_results=yes)
invertfit ("observations.obsfile", "configuration.mkconfig", "fitB.txt", "invB.txt", catalogs="catalogo.mkcatalog")
invertfit ("observations.obsfile", "config2.cfg", "fitE.txt", "invE1.txt", catalogs="catalogo.mkcatalog", errors="equations")
invertfit ("observations.obsfile", "config2.cfg", "fitE.txt", "invE2.txt", catalogs="catalogo.mkcatalog", errors="obserrors", print="BV,XV,mV", objects="standards")
invertfit ("obs2.txt", "configuration.mkconfig", "fitG.txt", "invG.txt", catalogs="catalogo.mkcatalog", objects="program")
invertfit ("obs2.txt", "configuration.mkconfig", "fitG.txt", "invG2.txt", errors="undefined")
logout
