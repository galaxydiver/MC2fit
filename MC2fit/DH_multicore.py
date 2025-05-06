import numpy as np
import copy
from multiprocessing import Pool
import time
from tqdm import tqdm

def multicore_run(loop_ftn, loop_len, Ncore=1, show_progress=10000,
                  use_try=False, use_zip=False, errorcode=np.nan,
                  show_summary=True,
                  record=False, record_fn='done.dat',
                  **kwargs):
    print("======== Multicore Briefing ========")
    print('● Multicore run for {} with Ncore'.format(loop_ftn.__name__), Ncore)
    print('● Using Try (Ignoring Errors):', use_try)

    global try_wrapper_function, wrapper_function
    ### =========== Wrapping ==========================
    def try_wrapper_function(i): ## Function with try --> ignore error
        if(show_progress>0):
            if((i%show_progress==0) & (i!=0)):
                print(i, "/", loop_len, "| Time : ", time.time()-start)
        try:
            results=loop_ftn(i)
            if(record): np.savetxt(record_fn, np.array(['done']), fmt="%s")
            return results

        except:
            return errorcode ## Error code
            print(">> Error! Loop:", i)

    def wrapper_function(i):  ## Function without try --> Stop when the function has an error
        if(show_progress>0):
            if((i%show_progress==0) & (i!=0)):
                print(i, "/", loop_len, "| Time : ", time.time()-start)
        results=loop_ftn(i)
        if(record): np.savetxt(record_fn, np.array(['done']), fmt="%s")
        return results

    ## ================== Main ==========================
    start=time.time()
    if(Ncore==1):
        results_array = np.full(loop_len, np.nan, dtype=object)
        for i in range (loop_len):
            if(use_try): results_array[i]=try_wrapper_function(i)
            else: results_array[i]=wrapper_function(i)
        print("Warning! This is a single core process!")

    else:
        with Pool(processes=int(Ncore)) as pool:
            if(use_zip):
                if(use_try): results_array=list(tqdm(zip(*pool.imap(try_wrapper_function, np.arange(loop_len))), total=loop_len))
                else: results_array=list(tqdm(zip(*pool.imap(wrapper_function, np.arange(loop_len))), total=loop_len))
            else:
                if(use_try): results_array=list(tqdm(pool.imap(try_wrapper_function, np.arange(loop_len)), total=loop_len))
                else: results_array=list(tqdm(pool.imap(wrapper_function, np.arange(loop_len)), total=loop_len))
        pool.close()
        pool.join()

    print("Done! | Time : ", time.time()-start)

    if(use_zip==False):
        if(show_summary):
            result=np.array(results_array).astype(float)
            Nfailed=np.sum(np.isnan(result))
            Nwarning=np.sum(result!=0)
            print("======== Multicore summary ========")
            print("● Succeed: %d / %d"%(np.sum(result==0), len(result)))
            print("● Failed: %d / %d"%(Nfailed, len(result)))
            print("● Warning (w/failed): %d / %d"%(Nwarning, len(result)))
            if(Nfailed>0):
                print(">> See where: self.multi_res")
                print(">> To debug, use Ncore=1, and use_try = False")
            print("==================================\n")

    return results_array
