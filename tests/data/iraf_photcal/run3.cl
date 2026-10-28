noao
digiphot
photcal
invertfit ("observations.obsfile", "configuration.mkconfig", "fit1.txt", "invP1.txt", catalogs="catalogo.mkcatalog", format="%-10s %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f\n")
invertfit ("observations.obsfile", "config2.cfg", "fitE.txt", "invPE1.txt", catalogs="catalogo.mkcatalog", errors="equations", format="%-10s %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f\n")
invertfit ("observations.obsfile", "config2.cfg", "fitE.txt", "invPE2.txt", catalogs="catalogo.mkcatalog", errors="obserrors", print="BV,XV,mV", objects="standards", format="%-10s %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f\n")
invertfit ("obs2.txt", "configuration.mkconfig", "fitG.txt", "invPG2.txt", errors="undefined", format="%-10s %14.7f %14.7f\n")
logout
