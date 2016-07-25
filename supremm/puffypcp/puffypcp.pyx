from pcp import pmapi
import cpmapi as c_pmapi
import numpy
from ctypes import c_uint

    
def getindomdict(ctx, metric_id_array):
    """ build a list of dicts that contain the instance domain id to text mappings
        The nth list entry is the nth metric in the metric_id_array
        @throw MissingIndomException if the instance information is not available
    """
    indomdict = []
    for i in xrange(len(metric_id_array)):
        metric_desc = ctx.pmLookupDesc(metric_id_array[i])
        if 4294967295 != pmapi.get_indom(metric_desc):
            try:
                ivals, inames = ctx.pmGetInDom(metric_desc)
                if ivals == None:
                    indomdict.append({})
                else:
                    indomdict.append(dict(zip(ivals, inames)))

            except pmapi.pmErr as exp:
                if exp.args[0] == c_pmapi.PM_ERR_INDOM:
                    indomdict.append({})
                elif exp.args[0] == c_pmapi.PM_ERR_INDOM_LOG:
                    return None
                else:
                    raise exp

        else:
            indomdict.append({})

    return indomdict

def loadrequiredmetrics(context, requiredMetrics):
    """ required metrics are those that must be present for the analytic to be run """
    try:
        required = context.pmLookupName(requiredMetrics)
        return [required[i] for i in xrange(0, len(required))]

    except pmapi.pmErr as e:
        if e.args[0] == c_pmapi.PM_ERR_NAME:
            # Required metric missing - this analytic cannot run on this archive
            return []
        else:
            raise e

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
    for i in xrange(0, len(metriclist)):
        metricarray[i] = metriclist[i]

    return metricarray

def getmetrictypes(context, metric_ids):
    """ returns a list with the datatype of the provided array of metric ids """
    return [context.pmLookupDesc(metric_ids[i]).type for i in xrange(len(metric_ids))]

def pcptypetonumpy(pcptype):
    """ Convert pcp data types to numpy equivalents """
    if pcptype == c_pmapi.PM_TYPE_STRING:
        return object
    return numpy.float

