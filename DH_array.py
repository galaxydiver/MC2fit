import numpy as np
import copy

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def ang_dist(r1, d1, r2, d2):
    """
    INPUT : RA1, Dec1, RA2, Dec2 in deg unit
    OUTPUT : Angular distance in deg unit
    """
    d2r=np.pi/180
    res=np.sin(d1*d2r)*np.sin(d2*d2r)+np.cos(d1*d2r)*np.cos(d2*d2r)*np.cos((r2-r1)*d2r)
    return np.arccos(res)/d2r

def dms2deg(dmsarray, hms=False, return_hms=False):
    dmsarray=np.array(dmsarray)
    if(np.ndim(dmsarray)==1):  # If input value = 1D array
        is_1darray=True
        dmsarray=np.array([dmsarray])
    else: is_1darray=False

    answer=dmsarray[:,0]+dmsarray[:,1]/60+dmsarray[:,2]/3600
    if(hms==True): answer=answer*360/24
    if(return_hms==True): answer=answer*24/360

    if(is_1darray==True): return answer[0] # If input value = 1D ndarray -> return single value
    else: return answer # return 1D array

def deg2dms(deg, hms=False, return_hms=False):
    deg=np.array(deg)
    if(np.ndim(deg)==0):  # If input value = just value, not ndarray
        is_justvalue=True
        deg=np.array([deg])
    else: is_justvalue=False
    if(hms==True): deg=deg*360/24

    if(return_hms==True): deg=deg/360*24
    answer=np.zeros((len(deg),3))
    answer[:,0]=deg.astype(int)
    answer[:,1]=(deg*60-answer[:,0]*60).astype(int)
    answer[:,2]=(deg*3600-answer[:,0]*3600 - answer[:,1]*60)

    if(is_justvalue==True): return answer[0] # If input value = just value, not ndarray -> return 1D array
    else: return answer # return 2D array


class ClassCut():
    def __init__(self, OriginalClass,
                 attrlist=None, cutlist=None
                ):

        self.OriginalClass=OriginalClass
        if((hasattr(attrlist, "__len__")) & (hasattr(cutlist, "__len__"))):
            self.cutlist=np.array(cutlist)
            for thisattr in attrlist:
                self.cutdata(thisattr)

    def cutdata(self, thisattr):
        thisattr=thisattr.split('&')
        prevdata=self.OriginalClass
        for i, attr in enumerate(thisattr):
            thisdata=getattr(prevdata, attr)
            if(i==len(thisattr)-1): ## Last
                thisdata=thisdata[self.cutlist]
                setattr(prevdata, attr, thisdata)
            else:
                prevdata=thisdata


class Flatten:
    """
    Descr - Use array_flatten in class type. (See array_flatten)
    INPUT
     - array - series of 2D arrays
     - *add_index - add an index showing ID of previous 2D array
    OUTPUT
     - self.data : 2D array
     - self.diff : starting point of each 2D array
     - self.ndata : total number of data
    Function
     - self.selectdata : choose one 2D array with ID

    example
    -------
    replace_field_name(gg.star, "time", "age")
    """
    def __init__(self, inputarray, add_index=False, debug=False):
        self.data, self.diff=array_flatten(inputarray, return_diff=True, add_index=add_index, debug=debug)
        self.ndata=len(self.diff)-1

    def selectdata(self, idnum):
        if(type(idnum)==int):
            idnum=int(idnum)
            return self.data[self.diff[int(idnum)]:self.diff[int(idnum+1)]]
        elif((type(idnum)==list) | (type(idnum)==np.ndarray)):
            return self.data[np.isin(self.data['array_index'], idnum)]


def array_flatten(array, return_diff=False, add_index=False, debug=False, return_in_object=False):
    """
    Descr - series of 2D arrays -> one 2D array
    INPUT
     - array - series of 2D arrays
     - *return_diff - return starting point of each 2D array   (default : False)
     - *add_index - add an index showing ID of previous 2D array
    OUTPUT
     - 2D array
     - *2D array, countarray (if return_diff == True)

    example
    -------
    """

    ## ======== Count & dtype ==========
    is_dtype_save=False
    countarray=np.zeros((len(array))).astype(int)  # The array for # of raws
    for i in range (len(array)):
        if(type(array[i])==np.ndarray):
            countarray[i]=len(array[i])
            if(is_dtype_save==False):
                array_dtype=array[i].dtype
                is_dtype_save=True
        else:
            countarray[i]=0
            if(debug==True): print("NOT ndarray at ", i)
    if(return_in_object): array_dtype=object
    countarray=np.cumsum(countarray)
    countarray=np.append([0], countarray)



    array_1D=np.zeros(countarray[-1], dtype=array_dtype)
    for i in range (len(array)):
        array_1D[countarray[i]:countarray[i+1]]=array[i]

    if(add_index==True):
        newdtype=[('array_index', '<i4')]+array_dtype.descr
        array_1D_add=np.zeros(countarray[-1], newdtype)
        for name in array_dtype.names:
            array_1D_add[name]=array_1D[name]
        for i in range (len(array)):
            array_1D_add[countarray[i]:countarray[i+1]]['array_index']=i
        array_1D=array_1D_add

    if(return_diff==True): return array_1D, countarray
    else: return array_1D


#==============================================================
def array_replace_field_name(arr, oldname, newname):
    """
    Descr - replace an old field name with a new field name.
    INPUT
     - array
     - oldname
     - newname
    OUTPUT
     - None  (input array will be changed using pointer)

    example
    -------
    replace_field_name(gg.star, "time", "age")
    """
    orgnames = list(arr.dtype.names)
    arr.dtype.names = tuple([newname if nn == oldname else nn for nn in orgnames])

def array_remove_field_name(a, remove_name):
    names = list(a.dtype.names)
    if remove_name in names:
        names.remove(remove_name)
    b = a[names]
    return b


def array_flexible_to_simple(array, dtype=None, remove_void=True):
    """
    Descr - structure array -> simple array (array without field names)
    ## from numpy.lib import recfunctions as rfn
    ** Oldname : array_remove_flexible
    """
    if(remove_void): array=array_remove_void(array)
    return np.lib.recfunctions.structured_to_unstructured(array, dtype=dtype, copy=True)



def array_quickname(array, dtype=None, names=None, dtypes=None, all_string=False):
    """
    Descr - simple 2D ndarray -> structure array (array with field names)
    INPUT
     - array
     - *dtype : input dtype (default : None)
        e.g.) [('col1', '<f4'), ('col2', '<f8')]
     - *names : namelist of columns    (default : None)
     - *dtypes : input dtype.          (default : None)
         e.g.) 'f4,f4,i4'
    OUTPUT
     - structure array
    """
    ## Check array = 2d array
    array=np.array(array)
    if(np.ndim(array)<2):
        print(">>> Error! Array should be a 2D array!")
        return

    ## If it has dtype
    if(hasattr(dtype, "__len__")):
        newarray=np.zeros(len(array), dtype=dtype)

    ## If names == None, generate a new array
    elif(type(names)==type(None)):
        if(type(dtypes)==type(None)):
            if(all_string): dtypes=['U20' for s in np.arange(len(array[0])).astype(str)]
            else: dtypes=['f8' for s in np.arange(len(array[0])).astype(str)]
        elif(dtypes==object):
            dtypes=[object for s in np.arange(len(array[0])).astype(str)]
        dtypes=list_to_string(dtypes)
        newarray=np.zeros(len(array), dtype=dtypes)

    ## If there are names, make newdtype
    else:
        names=string_to_list(names)
        if(type(dtypes)==type(None)):
            if(all_string): dtypes=['U20' for s in np.arange(len(array[0])).astype(str)]
            else: dtypes=['f8' for s in np.arange(len(array[0])).astype(str)]
        elif(dtypes==object):
            dtypes=[object for s in np.arange(len(array[0])).astype(str)]
        dtypes=string_to_list(dtypes)

        newdtype=[] #generate newdtype
        for i in range (len(array[0])):
            newdtype=newdtype+[(names[i], dtypes[i])]
            #prefix='autoid_'
            #namelist=[prefix + s for s in np.arange(len(array[0])).astype(str)]  # autoid_#
        newarray=np.zeros(len(array), dtype=newdtype)

    #============= new array =====================
    for i, name in enumerate(list(newarray.dtype.names)):
        newarray[name]=np.copy(array[:,i])
    return newarray





def array_quickname_old(array, dtype=None, names=None, dtypes=None):
    """
    Descr - simple 2D ndarray -> structure array (array with field names)
    INPUT
     - array
     - *dtype : input dtype (default : None)
        e.g.) [('col1', '<f4'), ('col2', '<f8')]
     - *names : namelist of columns    (default : None)
     - *dtypes : input dtype.          (default : None)
         e.g.) 'f4,f4,i4'
    OUTPUT
     - structure array
    """
    ## Check array = 2d array
    array=np.array(array)
    if(np.ndim(array)<2):
        print(">>> Error! Array should be a 2D array!")
        return

    ## If it has dtype
    if(hasattr(dtype, "__len__")):
        newarray=np.zeros(len(array), dtype=dtype)

    ## If names == None, generate a new array
    elif(type(names)==type(None)):
        if(type(dtypes)==type(None)):
            dtypes=['f8' for s in np.arange(len(array[0])).astype(str)]
        elif(dtypes==object):
            dtypes=[object for s in np.arange(len(array[0])).astype(str)]
        dtypes=list_to_string(dtypes)
        newarray=np.zeros(len(array), dtype=dtypes)

    ## If there are names, make newdtype
    else:
        names=string_to_list(names)
        if(type(dtypes)==type(None)):
            dtypes=['f8' for s in np.arange(len(array[0])).astype(str)]
        elif(dtypes==object):
            dtypes=[object for s in np.arange(len(array[0])).astype(str)]
        dtypes=string_to_list(dtypes)

        newdtype=[] #generate newdtype
        for i in range (len(array[0])):
            newdtype=newdtype+[(names[i], dtypes[i])]
            #prefix='autoid_'
            #namelist=[prefix + s for s in np.arange(len(array[0])).astype(str)]  # autoid_#
        newarray=np.zeros(len(array), dtype=newdtype)

    #============= new array =====================
    for i, name in enumerate(list(newarray.dtype.names)):
        newarray[name]=np.copy(array[:,i])
    return newarray

def array_remove_void_old(data, input_dtype=False):
    """
    Descr - remove void type field
    INPUT
     - data : array or dtype (if input data is dtype, use 'input_dtype')
     - input_dtype : if input data is a dtype.descr, not an array   (default : False)
    OUTPUT
     - data (array -> array, dtype.descr -> dtype.descr)
    """
    if(input_dtype==True): newdtype=data
    else: newdtype=data.dtype.descr

    removelist=[]
    for i in range (len(newdtype)):  # Find all void
        if(newdtype[i][1][:2]=='|V'): removelist.append(i)
    for i in removelist[::-1]: # remove in backward. If we remove column forward, [i] -> [i-1], complicated
        newdtype.remove(newdtype[i])

    if(input_dtype==True): return newdtype  # dtype -> dtype
    else:   #array -> array
        newarray=np.zeros(len(data), dtype=newdtype)
        for name in newarray.dtype.names:
            newarray[name]=np.copy(data[name])
        return newarray

def array_remove_void(data):
    if(type(data)==list): return data
    return np.lib.recfunctions.repack_fields(data)


def array_add_columns(arr_base, arr_add, add_front=False):
    """
    Descr - Combine two arrays having same length
    Tip - array_quickname would be useful for arr_add
    INPUT
     - arr_base : original array
     - arr_add : additional array
     - *dtype_add : dtype of arr_add     (default : arr_add.dtype.descr)
     - *add_front : attach the new array at the front of the base array    (defatult : False)
     - *remove_void : remove void columns     (default : True)
     OUTPUT
      - combined array
    """
    arr_base=copy.deepcopy(arr_base)
    arr_add=copy.deepcopy(arr_add)
    arr_base=array_remove_void(arr_base)
    arr_add=array_remove_void(arr_add)
    if(add_front==True): newdtype=arr_add.dtype.descr+arr_base.dtype.descr
    else: newdtype=arr_base.dtype.descr+arr_add.dtype.descr
    #if(remove_void==True): newdtype=array_remove_void(newdtype, input_dtype=True)

    newarray=np.zeros(len(arr_base), dtype=newdtype)
    for name in arr_base.dtype.names:
        newarray[name]=np.copy(arr_base[name])
    for name in arr_add.dtype.names:
        newarray[name]=np.copy(arr_add[name])

    return newarray

def array_attach_string(array, add, add_at_head=False):
    """
    Descr - Add a string to all components of the input array (or list)
    """
    add=repeat_except_array(str(add), len(array))
    output=np.zeros(len(array), dtype='object')
    for i in range (len(array)):
        if(add_at_head==False): output[i]=str(array[i])+str(add[i]) # Add at the tail
        else: output[i]=str(add[i])+str(array[i]) # Add at the head
    return list(output)


#======================== LIST ============================
def list_select_comp(inputlist, column):
    """
    Descr : select one column of list (like an array)
    """
    column=int(column)
    newlist=list(np.zeros(len(inputlist)))
    for i in range (len(inputlist)):
        newlist[i]=inputlist[i, column]
    return newlist

def list_to_string(array, raise_error=True):
    if(type(array)==str): return array
    elif(hasattr(array, "__len__")): return ','.join(np.array(array).astype(str))
    else:
        if(raise_error==False): return None
        else: raise Exception('ERROR! Check the input list!')

def string_to_list(array, raise_error=True):
    if(type(array)==str): return array.replace(',', ' ').split()
    elif(hasattr(array, "__len__")): return array
    else:
        if(raise_error==False): return None
        else: raise Exception('ERROR! Check the input string!')

def list_flatten(array):
    """
    Descr - list in list -> 1D array
    e.g.) INPUT : list_flatten([[1,2],[3],[4,5], None, 3, ['asdf']])
          OUTPUT : array([1, 2, 3, 4, 5, 'asdf'], dtype=object)
    """
    length=np.zeros(len(array))
    for i in range (len(array)):
        try: length[i]=len(array[i])
        except: pass
    length=np.cumsum(length)
    length=np.append([0], length).astype(int)
    returnarray=np.zeros(int(length[-1]), dtype='object')
    for i in range (len(array)):
        returnarray[length[i]:length[i+1]]=np.copy(array[i])
    return returnarray

#===================== Etc =================
def repeat_except_array(item, size):
    """
    Descr - repeat item if item is not array (or list)
     - e.g.) (1, 4) -> [1,1,1,1]
     - e.g.) ([1], 4) -> [1] ("Size is not enough!")
    INPUT
     - items
    """
    if((type(item)!=str) & hasattr(item, "__len__")):
        if(len(item)==size): return item
        elif(len(item) < size): print("Size is not enough!")
        else: return item[:size]
    else: return np.repeat(item, size)

def value_repeat_array(value,  repeat):
    if(hasattr(value, "__len__")==False):
        value=np.array([value]*repeat)
    else: value=np.array(value)
    if(len(value)==1): value=np.array([value[0]]*repeat)
    return value

def make_aperture_mask(imgsize, radius, center=-1):
    mask=np.ones((imgsize, imgsize), dtype=int)
    if(center<0): center=int(imgsize/2)
    mx, my=np.mgrid[0:imgsize,0:imgsize]
    check=np.where((mx-center)**2 + (my-center)**2 <= radius**2)
    mask[check]=0
    return mask

def inverse_var_weighted_mean(data, err):
    nonnan=np.where(np.isfinite(data) & np.isfinite(err))
    w = 1/(err**2)
    mean = np.sum(w[nonnan]*data[nonnan])/np.sum(w[nonnan])
    eom = (1/np.sum(w[nonnan]))**0.5
    return mean, eom

def inverse_var_weighted_std(data, err):
    # Weighted sample standard deviation

    nonnan=np.where(np.isfinite(data) & np.isfinite(err))

    w = 1/(err**2)
    mean, __ = inverse_var_weighted_mean(data, err)

    term1 = w[nonnan] * (data[nonnan] - mean)**2
    s = (np.sum(term1)/np.sum(w[nonnan]))**0.5

    print(s)
    N = len(data[nonnan])

    s_err = s / (2*(N-1))**0.5
    return s, s_err
