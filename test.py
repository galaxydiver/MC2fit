class ResPack:
    def __init__(self, proj_folder, runlist,
                 dir_work_list, comp_crit='RChi_100', show_progress=100, Ncore=1):

        if(Ncore==1):
            self.__getdata(proj_folder, runlist, dir_work_list, comp_crit, show_progress)
        else:
            self.__multicore_getdata(proj_folder, runlist, dir_work_list, comp_crit, show_progress, Ncore)

        self.onlyval, self.onlyerr=generate_onlyval(self, self.bestdat, index=self.bob_index)

    def __getdata(self, proj_folder, runlist, dir_work_list, comp_crit, show_progress):
        self.fulldat=np.zeros(len(dir_work_list), dtype=object) #Full data
        self.bestdat=np.zeros(len(dir_work_list), dtype=object) #Best submodels
        self.bob_index=np.zeros(len(dir_work_list)).astype(int) #Best of Best - index
        self.Nfile=np.zeros(len(dir_work_list)).astype(int) #Number of available files
        start=time.time()
        for i, dir_work in enumerate(dir_work_list):
            if(show_progress>0):
                if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)
            try:
                Res=dhgalfit.ResultGalfit(proj_folder=proj_folder, runlist=runlist,
                                          dir_work=dir_work,
                                          fna_load=None, fn_load=None,
                                          group_id=None, comp_crit=comp_crit,
                                          silent=True, ignore_no_files=True)
                self.fulldat[i]=Res.save_data(fna_output=None, return_array=True)
                ResBest=dhgalfit.ResultGalfitBest(Res, include_warn=False, comp_crit=comp_crit)
                self.bestdat[i]=ResBest.save_data(fna_output=None, return_array=True)
                self.bob_index[i]=ResBest.best[0] ## Save only the first value
                self.Nfile[i]=np.sum(Res.Data.is_file_exist)
            except:
                print("Result Error ", i)

    def __multicore_getdata(self, proj_folder, runlist, dir_work_list, comp_crit,
                          show_progress, Ncore):

        global sub_getdata
        def sub_getdata(i):
            if(show_progress>0):
                if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)
            try:
                Res=dhgalfit.ResultGalfit(proj_folder=proj_folder, runlist=runlist,
                                          dir_work=dir_work_list[i],
                                          fna_load=None, fn_load=None,
                                          group_id=None, comp_crit=comp_crit,
                                          silent=True, ignore_no_files=True)
                fulldat=Res.save_data(fna_output=None, return_array=True)
                ResBest=dhgalfit.ResultGalfitBest(Res, include_warn=False, comp_crit=comp_crit)
                bestdat=ResBest.save_data(fna_output=None, return_array=True)
                bob_index=ResBest.best[0] ## Save only the first value
                Nfile=np.sum(Res.Data.is_file_exist)
                return fulldat, bestdat, int(bob_index), int(Nfile)
            except:
                print("Result Error ", i)
                return 0,0,0,0

        start=time.time()
        pool = Pool(processes=int(Ncore))
        fulldat, bestdat, bob_index, Nfile = zip(*pool.map(sub_getdata, np.arange(len(dir_work_list))))
        print("Done! | Time : ", time.time()-start)
        pool.close()
        pool.join()
        self.fulldat=fulldat #Full data
        self.bestdat=bestdat #Best submodels
        self.bob_index=bob_index #Best of Best - index
        self.Nfile=Nfile #Number of available files


    def generate_onlyval(self, dataarray, index=None):
        dataarray=self.bestdat
        onlyval=np.zeros(len(dataarray), dtype=dataarray[0].dtype)
        onlyerr=np.zeros(len(dataarray), dtype=dataarray[0].dtype)

        for i in range (len(dataarray)):
            try:
                thisarray=dataarray[i]
                vallist=np.where(thisarray['datatype']==1)[0]
                errlist=np.where(thisarray['datatype']==3)[0]

                if(hasattr(index, "__len__")):
                    onlyval[i]=dataarray[i][vallist][index[i]]
                    onlyerr[i]=dataarray[i][errlist][index[i]]
                else:
                    onlyval[i]=dataarray[i][vallist]
                    onlyerr[i]=dataarray[i][errlist]
            except:
                print("Error ", i)
        return onlyval, onlyerr
        ## fulldat, bestdat, bob_index, bob_onlyval, bob_onlyerr, Nfile

    def convert_base_params_array(self):
        self.psfpos, self.base_params1_array, self.base_params2_array = dhgalfit.convert_base_params_array(self.bob_onlyval)
