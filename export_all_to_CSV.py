
sas_dataset_names = """
"Ssr_t_ssik","121.5GB","Table","","25Feb2016:21:45:00"
"Lpr_t_sksube","14.6GB","Table","","25Feb2016:00:31:33"
"Lpr_uaf_t_sksube","12.2GB","Table","","25Feb2016:00:33:21"
"Lpr_t_diag","9.6GB","Table","","25Feb2016:00:28:49"
"Lpr_t_adm","4.5GB","Table","","25Feb2016:00:26:42"
"Pat_t_diag_ny","3.0GB","Table","","24Feb2016:20:15:42"
"Famgen_fseid001620","2.1GB","Table","","25Feb2016:22:01:09"
"Lpr_t_bes","1.9GB","Table","","26Feb2016:11:09:03"
"Pat_t_rekv_ny","1.7GB","Table","","24Feb2016:18:21:46"
"Lpr_uaf_t_diag","1.6GB","Table","","25Feb2016:00:28:58"
"Lpr_t_sksopr","1.4GB","Table","","25Feb2016:00:29:14"
"Cpr3_t_adresse_hist","1.2GB","Table","","24Feb2016:16:48:22"
"Cpr3_t_person_lbnr","702.4MB","Table","","29Feb2016:09:45:40"
"Lpr_uaf_t_adm","648.4MB","Table","","25Feb2016:00:26:47"
"Cpr3_t_arkiv_adresse_hist","579.9MB","Table","","24Feb2016:17:24:40"
"Lpr_uaf_t_sksopr","541.9MB","Table","","25Feb2016:00:29:19"
"Fseid00001620_lbnr_key","524.0MB","Table","","24Feb2016:14:40:59"
"Fseid00001620_population","468.3MB","Table","","24Feb2016:14:41:15"
"Lpr_t_opr","367.0MB","Table","","25Feb2016:00:29:01"
"Cpr3_t_fodested","176.4MB","Table","","24Feb2016:15:03:49"
"Cpr3_t_civil_hist","140.3MB","Table","","24Feb2016:15:24:14"
"Cpr3_t_adresse","139.1MB","Table","","24Feb2016:15:35:16"
"Car_t_tumor","137.1MB","Table","","24Feb2016:17:28:13"
"Cpr3_t_civil","115.0MB","Table","","24Feb2016:15:13:44"
"Mfr_mfr","101.6MB","Table","","25Feb2016:21:57:06"
"Dar_t_dodsaarsag_1","95.3MB","Table","","24Feb2016:17:29:35"
"Mfr_t_lfoed","68.3MB","Table","","25Feb2016:21:46:27"
"Dar_t_dodsaarsag_2","40.6MB","Table","","24Feb2016:17:30:15"
"Ser_t_stamdata","15.6MB","Table","","24Feb2016:20:17:40"
"Ser_t_who","14.6MB","Table","","24Feb2016:20:16:51"
"Abr_t_abort","13.6MB","Table","","26Apr2016:11:23:36"
"Abr_t_lpr_abort","9.9MB","Table","","26Apr2016:11:23:51"
"Mfr_kejsersnit","7.8MB","Table","","25Feb2016:21:57:06"
"Lpr_uaf_t_opr","4.3MB","Table","","25Feb2016:00:29:01"
"Cpr3_t_separation_hist","3.8MB","Table","","24Feb2016:15:25:01"
"Cpr3_t_separation","640.0KB","Table","","24Feb2016:15:24:27"
"""

ommit_datasets = set(["Ssr_t_ssik", "Fseid00001620_lbnr_key", "Fseid00001620_population"])

for line in sas_dataset_names.split("\n"):
    line = line.strip()
    if line == "":
        continue

    words = line.split(",")
    dataset_name = words[0].strip("\"")

    if dataset_name in ommit_datasets:
        continue

    print("proc export data={} outfile='{}' dbms=csv replace;".format("F1620." + dataset_name, "F:\\Brugere\\fskPioDwo\\FSEID1620\\" + dataset_name + ".csv"))

print("run;")

print()

print("proc export data=In01620.LMS_LAEGEMIDDELOPLYSNINGER (keep=VNR ATC ATC1 VOLUME VOLTYPECODE VOLTYPETXT) outfile='V:\\Projekter\\FSEID00001620\\Piotr\\Data\\raw_csv\\LMS_LAEGEMIDDELOPLYSNINGER.csv' dbms=csv replace;")
print("proc export data=In01620.LMS_EPIKUR (keep=CPR_ENCRYPTED EKSD VNR) outfile='V:\\Projekter\\FSEID00001620\\Piotr\\Data\\raw_csv\\LMS_EPIKUR.csv' dbms=csv replace;")
print("RUN;") #Super important!!!! Duh


#.export of SSR:
#libname LIB "F:\Projekter\FSEID00001620\Data\24feb2016";
#proc print data=LIB.ssr_t_ssik(obs=100);
#run;
#proc export data=LIB.ssr_t_ssik (keep=lbnr V_HONUGE C_SPECIALE V_ANTYDEL C_YDELSESNR V_KONTAKT V_ALDER year) outfile='V:\Projekter\FSEID00001620\Piotr\Data\raw_csv\ssr_t_ssik.csv' dbms=csv replace;
#run;