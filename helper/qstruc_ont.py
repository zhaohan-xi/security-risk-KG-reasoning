# generating query structures for ontological reasoning


query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    # (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    # (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    # (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    # ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',

                    # (('e', ('r',)), ('e', ('r',))): '2i',
                    # ((('e', ('r',)), ('e', ('r',))), ('r',)): '2ip',
                }

# xi/xip
def gen_xi_xip(query_name_dict, low, high):
    for x in range(low, high):
        xi_name = str(x) + 'i'
        xip_name = str(x) + 'ip'
        xi_struc = tuple([('e', ('r',)) for _ in range(x)])
        xip_struc = (xi_struc, ('r',))
        query_name_dict[xi_struc] = xi_name
        query_name_dict[xip_struc] = xip_name

    name_query_dict = {value: key for key, value in query_name_dict.items()}
    return query_name_dict, name_query_dict

# npp.xi/xip
def gen_npp_xi_xip(query_name_dict, name_query_dict, n, low, high):
    assert n <= low
    for x in range(low, high):
        xi_name = '%dpp.%di' % (n, x)
        xip_name = '%dpp.%dip' % (n, x)
        
        base_xi_name = '%di' % x  # xi
        base_xi_struc = name_query_dict[base_xi_name]   

        xi_struc = list(base_xi_struc)
        for i in range(n):
            xi_struc[i] = ('e', ('r', 'r',))
        xi_struc = tuple(xi_struc)
        xip_struc = (xi_struc, ('r',))
        
        query_name_dict[xi_struc] = xi_name
        query_name_dict[xip_struc] = xip_name

    name_query_dict = {value: key for key, value in query_name_dict.items()}
    return query_name_dict, name_query_dict

# nppp.xi/xip
def gen_nppp_xi_xip(query_name_dict, name_query_dict, n, low, high):
    assert n <= low
    for x in range(low, high):
        xi_name = '%dppp.%di' % (n, x)
        xip_name = '%dppp.%dip' % (n, x)
        
        base_xi_name = '%dpp.%di' % (n, x)  # npp.xi
        base_xi_struc = name_query_dict[base_xi_name]   
        
        xi_struc = list(base_xi_struc)
        for i in range(n):
            xi_struc[i] = ('e', ('r', 'r', 'r',))
        xi_struc = tuple(xi_struc)
        xip_struc = (xi_struc, ('r',))
        
        query_name_dict[xi_struc] = xi_name
        query_name_dict[xip_struc] = xip_name

    name_query_dict = {value: key for key, value in query_name_dict.items()}
    return query_name_dict, name_query_dict

# nppp.mpp.xi/xip
def gen_nppp_mpp_xi_xip(query_name_dict, name_query_dict, n, m, low, high):
    assert n+m <= low
    for x in range(low, high):
        xi_name = '%dppp.%dpp.%di' % (n, m, x)
        xip_name = '%dppp.%dpp.%dip' % (n, m, x)
        
        base_xi_name = '%dpp.%di' % (n+m, x)  # (n+m)pp.xi
        base_xi_struc = name_query_dict[base_xi_name]   
        
        xi_struc = list(base_xi_struc)
        for i in range(n):
            xi_struc[i] = ('e', ('r', 'r', 'r',))
        xi_struc = tuple(xi_struc)
        xip_struc = (xi_struc, ('r',))
        
        query_name_dict[xi_struc] = xi_name
        query_name_dict[xip_struc] = xip_name

    name_query_dict = {value: key for key, value in query_name_dict.items()}
    return query_name_dict, name_query_dict

LOW, HIGH = 2, 15
query_name_dict, name_query_dict = gen_xi_xip(query_name_dict, LOW, HIGH)
query_name_dict, name_query_dict = gen_npp_xi_xip(query_name_dict, name_query_dict, 1, LOW, HIGH)
query_name_dict, name_query_dict = gen_npp_xi_xip(query_name_dict, name_query_dict, 2, LOW, HIGH)
query_name_dict, name_query_dict = gen_npp_xi_xip(query_name_dict, name_query_dict, 3, 3, HIGH)

query_name_dict, name_query_dict = gen_nppp_xi_xip(query_name_dict, name_query_dict, 1, LOW, HIGH)
query_name_dict, name_query_dict = gen_nppp_mpp_xi_xip(query_name_dict, name_query_dict, 1, 1, LOW, HIGH)
query_name_dict, name_query_dict = gen_nppp_mpp_xi_xip(query_name_dict, name_query_dict, 1, 2, 3, HIGH)
