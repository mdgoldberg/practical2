import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

def first_last_system_call_feats(tree, fn):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree, fn):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

def stringify(tree, fn):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c["stringified-"+el.tag] =ET.tostring(el)
    return c

def each_syscall_count(tree, fn):
    """
    gets the number of times each system call is called
    """
    counter = Counter()
    c = Counter()
    for all_sec in tree.iter('all_section'):
        for syscall in all_sec:
            counter['num-' + syscall.tag] += 1
    return counter


def num_processes(tree, fn):
    """
    gets the number of processes in the exe
    """
    c = Counter()
    for proc in tree.iter('process'):
        c['num_processes'] += 1
    return c

def num_threads(tree, fn):
    """
    gets the number of threads in the exe
    """
    c = Counter()
    for thread in tree.iter('thread'):
        c['num_threads'] += 1
    return c

def threads_per_process(tree, fn):
    c = Counter()
    lst = []
    for proc in tree.iter('proc'):
        tot = 0
        for thread in proc:
            tot += 1
        list.append(tot)
    mn = np.mean(lst) if lst else 1.0
    return {'threads_per_process': mn}

def percent_successful_syscalls(tree, fn):
    succ = 0
    tot = 0
    for all_sec in tree.iter('all_section'):
        for syscall in all_sec:
            if 'successful' not in syscall.attrib:
                continue
            if syscall.attrib['successful'] == "1":
                succ += 1
            tot += 1
    return {'percent_successful': succ/float(tot)}

# NOT BEING USED
def syscalls_per_thread_and_proc(tree, fn):
    num_threads = 0
    num_procs = 0
    num_syscalls = 0
    for proc in tree.iter('process'):
        num_procs += 1
        for thread in proc.iter('thread'):
            num_threads += 1
            if len(thread):
                for syscall in thread.find('all_section'):
                    num_syscalls += 1
    return {'syscalls_per_thread': float(num_syscalls)/num_threads if
            num_threads else num_syscalls, 'syscalls_per_proc':
            float(num_syscalls)/num_procs if num_procs else num_syscalls}


def load_dll_files(tree, fn):
    c = Counter()
    for all_sec in tree.iter('all_section'):
        for syscall in all_sec:
            if syscall.tag == 'load_dll' and 'filename' in syscall.attrib:
                c['load_dll-'+syscall.attrib['filename']] += 1
    return c

def vm_protect_targets(tree, fn):
    c = Counter()
    for all_sec in tree.iter('all_section'):
        for syscall in all_sec:
            if syscall.tag == 'vm_protect' and 'target' in syscall.attrib:
                c['vm_protect-'+syscall.attrib['target']] += 1
    return c

ffs = [first_last_system_call_feats, system_call_count_feats,
        each_syscall_count, num_processes, num_threads, threads_per_process,
        percent_successful_syscalls, load_dll_files, vm_protect_targets]
