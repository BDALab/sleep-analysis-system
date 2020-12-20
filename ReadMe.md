# Sleep analysis system
A web system created to process and analyze data from actigraphy to distinguish sleep patterns
and identify sleep disorders.

# How to make this project work on your machine

- Download this project from github.com (git required on your PC)

    `git clone https://github.com/BDALab/sleep-analysis-system.git`

- Create the environment with Python packages from **environment.yml** (Anaconda required)

    - https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file

- Setup your database according to https://docs.djangoproject.com/en/3.1/ref/databases/

- Type this command into your command line with conda environment created and activated, you must be in **www/mysite** folder where the **manage.py** file is located:

    `python manage.py runserver`

    This command will start http server, for testing. If you want to run https server, type:

    `python manage.py runserver_plus`
    
    You will need a certifivate if you want to run https server, take a look at https://django-extensions.readthedocs.io/en/latest/runserver_plus.html

## Dataset
There are those features extracted in the project, so you can learn your own model etc.
But if you would like to test preprocessing and other stuffs like that, you need to 
download the dataset from https://doi.org/10.5281/zenodo.1160410.

Then you need to convert the data from binary into csv using **GENEActivPcSoftware**,
you can download it from https://www.activinsights.com/resources-support/geneactiv/downloads-software/. The converted csv can be uploaded on the server as **csv data**,
the data from polysomnography can be uploaded into **polysomnography data** (txt files).



