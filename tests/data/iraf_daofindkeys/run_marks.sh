# IRAF 2.18 marking what daofind finds: sh run_marks.sh NAME IMAGE KEYS, KEYS being NAME.cur
# with the answers to q after it. With a cursor file IRAF marks nothing, and it wants the
# display to place the marks, so ds9 has to be up (ds9 -iconify yes -unix /tmp/.IMT1000
# -fifo none -port none, under a timeout) and the plain CL reads the image cursor as text.
# The marks go to home/.iraf/uparm/vdm, for ../iraf_photcal/decode_gki.py --everything; HOME
# is a scratch one, so that nothing is saved in the account's uparm.
rm -rf home/.iraf/uparm/*; rm -f $name.coo* $name.gki
{ cat <<EOT
images
tv
set stdimage = imt512
set stdimcur = "text"
display $image 1 fill=no >& display.log
noao
digiphot
apphot
datapars.fwhmpsf = 2.5
datapars.sigma = 6.
datapars.readnoise = 0.
datapars.epadu = 1.
findpars.threshold = 4.
findpars.nsigma = 1.5
findpars.ratio = 1.
findpars.theta = 0.
findpars.sharplo = 0.2
findpars.sharphi = 1.
findpars.roundlo = -1.
findpars.roundhi = 1.
daofind $image output=$name.coo interactive=yes icommands="" verify=no update=no verbose=no starmap="" skymap="" display=stdvdm graphics=stdplot mkdetections=yes
EOT
cat $keys
echo logout; } > $name.cl
(ulimit -f 50000; HOME=$PWD/home timeout 30 irafcl < $name.cl > $name.clout 2>&1)
