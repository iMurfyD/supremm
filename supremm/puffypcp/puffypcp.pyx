from pcp import pmapi
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import cpmapi as c_pmapi
import numpy
from ctypes import c_uint

cimport pcp
cimport numpy

cdef extern from "Python.h":
    ctypedef struct Py_buffer:
        void* buf # Not sure if that's actual implementaion
    int PyBUF_SIMPLE
    int PyObject_GetBuffer(object, Py_buffer*, int)  
    ctypedef void PyObject
    PyObject* PyLong_FromLong(long)
    PyObject* PyLong_FromUnsignedLong(unsigned long)
    PyObject* PyLong_FromLongLong(long long)
    PyObject* PyLong_FromUnsignedLongLong(unsigned long long)

# Just so it will compile - not sure if these are the acutal definitions
cdef extern from "inttypes.h":
    ctypedef intptr_t 
    ctypedef int int32_t 
    ctypedef unsigned int uint32_t 
    ctypedef long int64_t 
    ctypedef unsigned long uint64_t 

cdef object topyobj(pcp.pmAtomValue atom, int dtype):
    if dtype == pcp.PM_TYPE_STRING:
        return str(atom.cp)
    elif dtype == pcp.PM_TYPE_32:
        return long(atom.l)
    elif dtype == pcp.PM_TYPE_U32:
        return long(atom.ul)
    elif dtype == pcp.PM_TYPE_64:
        return long(atom.ll)
    elif dtype == pcp.PM_TYPE_U64:
        return long(atom.ull)
    elif dtype == pcp.PM_TYPE_DOUBLE:
        return long(atom.d)
    else: # Don't know how to handle data type
        print "Unknown data type"
        return long(atom.cp)

cdef object strinnerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef pcp.pmAtomValue atom
    cdef int status
    tmp_data = list()
    for j in xrange(numval):
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_STRING, &atom, pcp.PM_TYPE_STRING)
        if status < 0:
            raise pmapi.pmErr(status)     
        tmp_data.append(str(atom.cp))
    return numpy.array(tmp_data)

cdef numpy.ndarray[double, ndim=1, mode="c"] int32innerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef pcp.pmAtomValue atom
    cdef int status
    cdef numpy.ndarray[double, ndim=1, mode="c"] tmp_data = numpy.empty(numval, dtype=numpy.float64)
    tmp_data = tmp_data # To update cython reference
    for j in xrange(numval):
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_32, &atom, pcp.PM_TYPE_32)
        if status < 0:
            raise pmapi.pmErr(status)     
        tmp_data[j] = <double>atom.l
    return numpy.array(tmp_data)

cdef numpy.ndarray[double, ndim=1, mode="c"] uint32innerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef int status
    cdef pcp.pmAtomValue atom
    cdef numpy.ndarray[double, ndim=1, mode="c"] tmp_data = numpy.empty(numval, dtype=numpy.float64)
    tmp_data = tmp_data
    for j in xrange(numval):
        inst = res.vset[i].vlist[j].inst 
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_U32, &atom, pcp.PM_TYPE_U32)
        if status < 0:
            raise pmapi.pmErr(status)     
    return numpy.array(tmp_data)

cdef numpy.ndarray[double, ndim=1, mode="c"] int64innerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef pcp.pmAtomValue atom
    cdef int status
    cdef numpy.ndarray[double, ndim=1, mode="c"] tmp_data = numpy.empty(numval, dtype=numpy.float64)
    tmp_data = tmp_data
    for j in xrange(numval):
        inst = res.vset[i].vlist[j].inst 
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_64, &atom, pcp.PM_TYPE_64)
        if status < 0:
            raise pmapi.pmErr(status)     
        tmp_data[j] = <double>atom.ll
    return tmp_data

cdef numpy.ndarray[double, ndim=1, mode="c"] uint64innerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef pcp.pmAtomValue atom
    cdef int status
    cdef numpy.ndarray[double, ndim=1, mode="c"] tmp_data = numpy.empty(numval, dtype=numpy.float64)
    tmp_data = tmp_data
    for j in xrange(numval):
        inst = res.vset[i].vlist[j].inst 
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_U64, &atom, pcp.PM_TYPE_U64)
        if status < 0:
            raise pmapi.pmErr(status)     
        tmp_data[j] = <double>atom.ull
    return numpy.array(tmp_data)

cdef numpy.ndarray[double, ndim=1, mode="c"] doubleinnerloop(int numval, pcp.pmResult* res, int i):
    cdef Py_ssize_t j
    cdef pcp.pmAtomValue atom
    cdef int status
    cdef numpy.ndarray[double, ndim=1, mode="c"] tmp_data = numpy.empty(numval, dtype=numpy.float64)
    cdef double* tmp_datap = &tmp_data[0]
    for j in xrange(numval):
        inst = res.vset[i].vlist[j].inst 
        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], pcp.PM_TYPE_DOUBLE, &atom, pcp.PM_TYPE_DOUBLE)
        if status < 0:
            raise pmapi.pmErr(status)     
        tmp_datap[j] = atom.d
    return tmp_data

# All numeric types return numpy.float64 (c double) arrays
# Functions are seperated based on type to handle any quirks from converting to double
cdef object extractValuesInnerLoop(Py_ssize_t numval, pcp.pmResult* res, int dtype, int i):
    if dtype == pcp.PM_TYPE_STRING:
        return strinnerloop(numval, res, i) 
    elif dtype == pcp.PM_TYPE_32:
        return int32innerloop(numval, res, i)
    elif dtype == pcp.PM_TYPE_U32:
        return uint32innerloop(numval, res, i)
    elif dtype == pcp.PM_TYPE_64:
        return int64innerloop(numval, res, i)
    elif dtype == pcp.PM_TYPE_U64:
        return uint64innerloop(numval, res, i)
    elif dtype == pcp.PM_TYPE_DOUBLE:
        return doubleinnerloop(numval, res, i)
    else: # Don't know how to handle data type
        print "Don't know how to handle data type"
        return []

cdef char* lookup(int val, int len, int* instlist, char** namelist):
    cdef int i
    for i in xrange(len):
        if instlist[i] == val:
            return namelist[i]
    return NULL

def extractValues(context, result, py_metric_id_array, mtypes):
    data = []
    description = []

    cdef Py_buffer buf
    PyObject_GetBuffer(result.contents, &buf, PyBUF_SIMPLE)
    cdef pcp.pmResult* res = <pcp.pmResult*> buf.buf
    cdef int ninstances
    cdef int numpmid = res.numpmid
    cdef Py_ssize_t i, j, k
    cdef int ctx = context._ctx
    cdef int status, inst
    cdef int* ivals
    cdef char** inames
    cdef char* name
    cdef pcp.pmDesc metric_desc
    cdef pcp.pmAtomValue atom
    cdef int dtype

    if numpmid < 0:
        return None, None

    cdef pcp.pmID* metric_id_array = <pcp.pmID*>PyMem_Malloc(numpmid * sizeof(pcp.pmID))
    for i in xrange(numpmid):
        metric_id_array[i] = py_metric_id_array[i] # Implicit py object to c data type conversion
    pcp.pmUseContext(ctx)

    for i in xrange(numpmid):
        ninstances = res.vset[i].numval
        ninstances = ninstances
        if ninstances < 0:
            PyMem_Free(metric_id_array)
            return None, None
        # No instances, but there needs to be placeholders
        elif ninstances == 0:
            data.append(numpy.empty(0, dtype=numpy.float64))
            description.append([numpy.empty(0, dtype=numpy.int64), []])
        else:
            dtype = mtypes[i]       
            tmp_names = []
            tmp_idx = numpy.empty(ninstances, dtype=int)

            # extractValueInneLoop does own looping 
            data.append(extractValuesInnerLoop(ninstances, res, dtype, i))
            status = pcp.pmLookupDesc(metric_id_array[i], &metric_desc) 
            if status < 0:
                PyMem_Free(metric_id_array)
                return None, None
            status = pcp.pmGetInDom(metric_desc.indom, &ivals, &inames)
            if status < 0:
                if len(data[i]) != 0: # Found data, so insert placeholder description
                    description.append([numpy.empty(0, dtype=numpy.int64), []])
                else: 
                    PyMem_Free(metric_id_array)
                    return None, None
            elif ninstances > status: # Missing a few indoms - skip 
                PyMem_Free(metric_id_array)
                return True, True
                #data.append(numpy.empty(0, dtype=numpy.float64))
                #description.append([numpy.empty(0, dtype=numpy.int64), []])
            else: 
                if ninstances > status:
                    pass # Add logging here 
                for j in xrange(ninstances):
                    if res.vset[i].vlist[j].inst == 4294967295:
                        pass # Add logging here
                    tmp_idx[j] = res.vset[i].vlist[j].inst
                    # TODO - find way to just look for one name not generate list then find it in list
                    name = lookup(res.vset[i].vlist[j].inst, status, ivals, inames)          
                    if name == NULL:
                        pass # Possibly add logging here
                    tmp_names.append(name)   
                        
                description.append([tmp_idx, tmp_names])
 
                PyMem_Free(ivals)
                PyMem_Free(inames)

    PyMem_Free(metric_id_array)

    return data, description

def extractpreprocValues(context, result, py_metric_id_array, mtypes):
    data = []
    description = []

    cdef Py_buffer buf
    PyObject_GetBuffer(result.contents, &buf, PyBUF_SIMPLE)
    cdef pcp.pmResult* res = <pcp.pmResult*> buf.buf
    cdef int mid_len = len(py_metric_id_array)
    cdef int numpmid = res.numpmid
    cdef int ninstances
    cdef Py_ssize_t i, j
    cdef int ctx = context._ctx
    cdef int status
    cdef int* ivals
    cdef char** inames
    cdef pcp.pmDesc metric_desc
    cdef pcp.pmAtomValue atom
    cdef int dtype

    if mid_len < 0:
        return None, None
    cdef pcp.pmID* metric_id_array = <pcp.pmID*>PyMem_Malloc(mid_len * sizeof(pcp.pmID))
   
    for i in xrange(mid_len):
        metric_id_array[i] = py_metric_id_array[i] # Implicit py object to c data type conversion
    pcp.pmUseContext(ctx)

    # Initialize description
    for i in xrange(mid_len):
        pcp.pmLookupDesc(metric_id_array[i], &metric_desc)
        if 4294967295 == metric_desc.indom: # Missing indom - skip
            continue 
        status = pcp.pmGetInDom(metric_desc.indom, &ivals, &inames)
        if status < 0:
            description.append({})
        else:
            tmp_dict = dict()
            for j in xrange(status):
                tmp_dict[ivals[j]] = inames[j]
            description.append(tmp_dict)         
            PyMem_Free(ivals)
            PyMem_Free(inames)

    # Initialize data
    for i in xrange(numpmid):
        ninstances = res.vset[i].numval
        ninstances = ninstances
        pcp.pmLookupDesc(metric_id_array[i], &metric_desc)

        tmp_data = []
        dtype = mtypes[i]

        for j in xrange(ninstances):
            status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], dtype, &atom, dtype)
            if status < 0:
                print "Couldn't extract value"
                tmp_data.append([])
            else:
                tmp_data.append([topyobj(atom, dtype), res.vset[i].vlist[j].inst])
    
        data.append(tmp_data)

    PyMem_Free(metric_id_array)
    return data, description
 
def getindomdict(context, py_metric_id_array):
    """ build a list of dicts that contain the instance domain id to text mappings
        The nth list entry is the nth metric in the metric_id_array
        @throw MissingIndomException if the instance information is not available
    """
    cdef int mid_len = len(py_metric_id_array)
    cdef pcp.pmID* metric_id_array = <pcp.pmID*>PyMem_Malloc(mid_len * sizeof(pcp.pmID))
    cdef Py_ssize_t i, j
    for i in xrange(mid_len):
        metric_id_array[i] = py_metric_id_array[i]
    cdef int ctx = context._ctx
    pcp.pmUseContext(ctx)
 
    indomdict = []
    cdef pcp.pmDesc metric_desc
    cdef int status
    cdef int* ivals
    cdef char** inames
    for i in xrange(mid_len):
        pcp.pmLookupDesc(metric_id_array[i], &metric_desc)
        if 4294967295 != metric_desc.indom:
            status = pcp.pmGetInDom(metric_desc.indom, &ivals, &inames)
            if status < 0: # TODO - add specific responses for different errors
                indomdict.append({})
            else:
                tmp_dict = dict()
                for j in xrange(status):
                    tmp_dict[ivals[j]] = inames[j]
                indomdict.append(tmp_dict)
                PyMem_Free(ivals)
                PyMem_Free(inames)
        else:
            indomdict.append({})

    PyMem_Free(metric_id_array)
    return indomdict

def loadrequiredmetrics(context, requiredMetrics):
    """ required metrics are those that must be present for the analytic to be run """
    cdef int num_met = len(requiredMetrics)
    cdef int ctx = context._ctx 
    pcp.pmUseContext(ctx)
    cdef Py_ssize_t i
    cdef int status
    cdef char** nameofmetrics = <char**>PyMem_Malloc(num_met * sizeof(char*))
    for i in xrange(num_met):
        nameofmetrics[i] = requiredMetrics[i]
    
    cdef pcp.pmID* required = <pcp.pmID*>PyMem_Malloc(num_met * sizeof(pcp.pmID*))
    status = pcp.pmLookupName(num_met, nameofmetrics, required)
    PyMem_Free(nameofmetrics)
    if status < 0: # Add specific error messages
        PyMem_Free(required)
        return []
        # Required metric missing - this analytic cannot run on this archive
    if status != num_met:
        PyMem_Free(required)
        return []
 
    ret = []
    for i in xrange(num_met):
        ret.append(required[i]) 
    PyMem_Free(required)

    return ret

def getmetricstofetch(context, analytic):
    """ returns the c_type data structure with the list of metrics requested
        for the analytic """

    metriclist = []

    for derived in analytic.derivedMetrics:
        context.pmRegisterDerived(derived['name'], derived['formula'])
        required = context.pmLookupName(derived['name'])
        metriclist.append(required[0])

    if len(analytic.requiredMetrics) > 0:
        metricOk = False
        if isinstance(analytic.requiredMetrics[0], basestring):
            r = loadrequiredmetrics(context, analytic.requiredMetrics)
            if len(r) > 0:
                metriclist += r
                metricOk = True
        else:
            for reqarray in analytic.requiredMetrics:
                r = loadrequiredmetrics(context, reqarray)
                if len(r) > 0:
                    metriclist += r
                    metricOk = True
                    break

        if not metricOk:
            return []

    for optional in analytic.optionalMetrics:
        try:
            opt = context.pmLookupName(optional)
            metriclist.append(opt[0])
        except pmapi.pmErr as e:
            if e.args[0] == c_pmapi.PM_ERR_NAME or e.args[0] == c_pmapi.PM_ERR_NONLEAF:
                # Optional metrics are allowed to not exist
                pass
            else:
                raise e


    metricarray = (c_uint * len(metriclist))()
    cdef Py_ssize_t i
    for i in xrange(0, len(metriclist)):
        metricarray[i] = metriclist[i]

    return metricarray

def getmetrictypes(context, py_metric_ids):
    """ returns a list with the datatype of the provided array of metric ids """
    cdef int num_mid = len(py_metric_ids)
    cdef Py_ssize_t i
    cdef pcp.pmID* metric_ids = <pcp.pmID*>PyMem_Malloc(num_mid * sizeof(pcp.pmID))
    for i in xrange(num_mid):
        metric_ids[i] = py_metric_ids[i]
    cdef int ctx = context._ctx
    pcp.pmUseContext(ctx)
    cdef pcp.pmDesc d
    cdef int ty
    metrictypes = list()
    for i in xrange(num_mid):
        pcp.pmLookupDesc(metric_ids[i], &d) 
        ty = d.type
        metrictypes.append(ty)

    PyMem_Free(metric_ids)
    return metrictypes

def pcptypetonumpy(pcptype):
    """ Convert pcp data types to numpy equivalents """
    if pcptype == pcp.PM_TYPE_STRING:
        return object
    return numpy.float

