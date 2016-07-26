from pcp import pmapi
from libc.stdlib cimport malloc, free
import cpmapi as c_pmapi
import numpy
import ctypes
from ctypes import c_uint
from ctypes import addressof

cimport pcp

cdef extern from "Python.h":
    ctypedef struct Py_buffer:
        void* buf # Not sure if that's actual implementaion
    int PyBUF_SIMPLE
    int PyObject_GetBuffer(object, Py_buffer*, int)  

cdef extern from "inttypes.h":
    ctypedef intptr_t

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
    else: # Don't know how to handle data type
        return long(atom.cp)

def extractpreprocValues(context, result, py_metric_id_array, mtypes):
    data = []
    description = []
   
    cdef Py_buffer buf
    PyObject_GetBuffer(result.contents, &buf, PyBUF_SIMPLE)
    cdef pcp.pmResult* res = <pcp.pmResult*> buf.buf
    cdef int mid_len = len(py_metric_id_array)
    cdef pcp.pmID* metric_id_array = <pcp.pmID*>malloc(mid_len * sizeof(pcp.pmID))
    cdef Py_ssize_t i, j, k
    cdef int ctx = context._ctx
    cdef int status, inst
    cdef int* ivals
    cdef char** inames
    cdef pcp.pmDesc metric_desc
    cdef pcp.pmAtomValue atom
    cdef int dtype
    for i in xrange(mid_len):
        metric_id_array[i] = py_metric_id_array[i] # Implicit py object to c data type conversion
    pcp.pmUseContext(ctx)
      
    for i in xrange(mid_len):
        pcp.pmLookupDesc(metric_id_array[i], &metric_desc) 
        if 4294967295 != metric_desc.indom:
            status = pcp.pmGetInDom(metric_desc.indom, &ivals, &inames)
            if status < 0: # TODO - add specific responses for different errors
                description.append({})
                data.append(numpy.array([]))
            else:
                tmp_dict = dict()
                tmp_data = []
                dtype = mtypes[i] 

                for j in xrange(status):
                    tmp_dict[ivals[j]] = inames[j]
                    if res.vset[i].numval > 0:
                        inst = res.vset[i].vlist[j].inst 
                        status = pcp.pmExtractValue(res.vset[i].valfmt, &res.vset[i].vlist[j], dtype, &atom, dtype)
                        if status < 0:
                            print "Couldn't extract value"
                            return [], []
                        tmp_data.append(topyobj(atom, dtype))

                description.append(tmp_dict)
                data.append(numpy.array(tmp_data))
                free(ivals)
                free(inames)
        else:
            description.append({})
            data.append(numpy.array([]))

    free(metric_id_array)


    return data, description
 
def getindomdict(context, py_metric_id_array):
    """ build a list of dicts that contain the instance domain id to text mappings
        The nth list entry is the nth metric in the metric_id_array
        @throw MissingIndomException if the instance information is not available
    """
    cdef int mid_len = len(py_metric_id_array)
    cdef pcp.pmID* metric_id_array = <pcp.pmID*>malloc(mid_len * sizeof(pcp.pmID))
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
                free(ivals)
                free(inames)
        else:
            indomdict.append({})

    free(metric_id_array)
    return indomdict

def loadrequiredmetrics(context, requiredMetrics):
    """ required metrics are those that must be present for the analytic to be run """
    cdef int num_met = len(requiredMetrics)
    cdef int ctx = context._ctx 
    pcp.pmUseContext(ctx)
    cdef Py_ssize_t i
    cdef int status
    cdef char** nameofmetrics = <char**>malloc(num_met * sizeof(char*))
    for i in xrange(num_met):
        nameofmetrics[i] = requiredMetrics[i]
    
    cdef pcp.pmID* required = <pcp.pmID*>malloc(num_met * sizeof(pcp.pmID*))
    status = pcp.pmLookupName(num_met, nameofmetrics, required)
    if status < 0: # Add specificc error messages
        free(required)
        return []
        # Required metric missing - this analytic cannot run on this archive
    if status != num_met:
        free(required)
        return []
 
    ret = []
    for i in xrange(num_met):
        ret.append(required[i]) 
    free(required)

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
    cdef pcp.pmID* metric_ids = <pcp.pmID*>malloc(num_mid * sizeof(pcp.pmID))
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

    free(metric_ids)
    return metrictypes

def pcptypetonumpy(pcptype):
    """ Convert pcp data types to numpy equivalents """
    if pcptype == pcp.PM_TYPE_STRING:
        return object
    return numpy.float

