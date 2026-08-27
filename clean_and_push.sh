cd /Volumes/Portable/geneactiv-processing-data
find .git -name '._*' -type f -delete
COPYFILE_DISABLE=1 git push origin master
COPYFILE_DISABLE=1 git push github master
