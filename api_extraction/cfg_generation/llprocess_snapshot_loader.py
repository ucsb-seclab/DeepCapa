"""
This module allows importing Lastline full-process snapshots into IDA Pro.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

:Copyright:
     Copyright 2014 Lastline, Inc.  All Rights Reserved.
"""
__author__ = 'Lastline, Inc., Saastha Vasan'

import gzip
import json
import os
from os import path
from os import walk
import struct
import tempfile
import tarfile
import sys
# we do our tests without IDA
#pylint: disable=F0401
import ida_idaapi
import ida_name
import ida_idp
import ida_kernwin
import ida_bytes
import ida_entry
import ida_xref
import idaapi
import idautils
import idc
import ida_pro
from idc import *

from cfg_extractor import cfg_extractor
import api_extraction.mongo_db.producer as producer
import time
#pylint: enable=F0401
# ignore references to IDA-defines values
#pylint: disable=E0602

PROCESS_DUMP_VERSION = 3
CODEHASH_VERSION = 4
MAX_REF_LEVEL = 10

#Defining constants
FF_BYTE = 0x00000000
# memory block access
PAGE_NOACCESS = 0x01     # winnt
PAGE_READONLY = 0x02     # winnt
PAGE_READWRITE = 0x04     # winnt
PAGE_WRITECOPY = 0x08     # winnt
PAGE_EXECUTE = 0x10     # winnt
PAGE_EXECUTE_READ = 0x20     # winnt
PAGE_EXECUTE_READWRITE = 0x40     # winnt
PAGE_EXECUTE_WRITECOPY = 0x80     # winnt
PAGE_GUARD = 0x100     # winnt
PAGE_NOCACHE = 0x200     # winnt
PAGE_WRITECOMBINE = 0x400     # winnt

# pe section cheracteristics
IMAGE_SCN_LNK_NRELOC_OVFL = 0x01000000  # Section contains extended relocations.
IMAGE_SCN_MEM_DISCARDABLE = 0x02000000  # Section can be discarded.
IMAGE_SCN_MEM_NOT_CACHED = 0x04000000  # Section is not cachable.
IMAGE_SCN_MEM_NOT_PAGED = 0x08000000  # Section is not pageable.
IMAGE_SCN_MEM_SHARED = 0x10000000  # Section is shareable.
IMAGE_SCN_MEM_EXECUTE = 0x20000000  # Section is executable.
IMAGE_SCN_MEM_READ = 0x40000000  # Section is readable.
IMAGE_SCN_MEM_WRITE = 0x80000000  # Section is writeable.

api_viewer = None

IDA_BITSIZE_32 = 32
IDA_BITSIZE_64 = 64



def va_from_string(value):
    return int(value.rstrip("L"), 16)


def make_func_name(va, name, flags = (idc.SN_NOCHECK | idc.SN_NOWARN)):
    cur_name = name
    #while not idc.MakeNameEx(va, str(cur_name), flags):
    while not idc.set_name(va, str(cur_name), flags):
        cur_name += '_'
        if len(cur_name) > 50:
            return None

    return cur_name

##define dr_O    1                       // Offset
#define fl_CF   16              // Call Far
#define fl_CN   17              // Call Near
#define fl_JF   18              // Jump Far
#define fl_JN   19              // Jump Near

def check_prev_reference(va, delta, va_set, level):

    if level >= 1:
        return 0

    level += 1
    xrefs_count = 0
    for xref in idautils.XrefsTo(va, 0):
        if xref.type not in [idc.dr_O, idc.fl_CF, idc.fl_CN, idc.fl_JF, idc.fl_JN]:
            continue

        xrefs_count += 1
        if xref.type == idc.dr_O:
            if sum(1 for _ in idautils.XrefsTo(xref.frm, 0)) >= 1:
                va_set.add(xref.frm)
                check_prev_reference(xref.frm, 0, va_set, level)

        else:
            cur_dref = ida_xref.get_first_dref_from(xref.frm)
            while cur_dref != 0xffffffff and cur_dref != 0xffffffffffffffff:
                va_set.add(cur_dref)
                cur_dref = ida_xref.get_next_dref_from(xref.frm, cur_dref)

        if not check_prev_reference(xref.frm, delta, va_set, level):
            if xref.type in [idc.fl_JF, idc.fl_JN] and delta:
                if check_prev_reference(xref.frm - delta, 0, va_set, level):
                    va_set.add(xref.frm - delta)

    return xrefs_count


def analyze_api_refs(dump_info):
    loaded_libraries = dump_info.get("loaded_libraries", None)
    if loaded_libraries is None:
        return

    api_log = []
    for dll in loaded_libraries:
        if 'exports' not in dll:
            continue
        base_addr = dll['virtual_address']
        dll_name = dll['object_description']
        image_base = va_from_string(dll['virtual_address'])

        lib_exports = {}
        for export in dll['exports'].values():
            export_va = image_base + export['rva']
            lib_exports["0x%x" % export_va] = export

        seg_xrefs = get_seg(base_addr)
        if seg_xrefs:
            for ref in seg_xrefs:
                # due to ASLR the address is stored in the JSON
                # as api_address - base_address.
                correct_ref = ref
                delta = 0
                while ("0x%x" % (ref - delta)) not in lib_exports and delta < 20:
                    delta += 1

                correct_ref = ref - delta
                correct_ref_str = "0x%x" % correct_ref
                if correct_ref_str in lib_exports:

                    name = lib_exports[correct_ref_str].get('name', None)
                    if name == None:
                        continue

                    refs_va = set()
                    check_prev_reference(ref, delta, refs_va, 0)
                    for va in refs_va:
                        make_func_name(va, name)
                        #idc.AddEntryPoint(va, va, str(name), 0)
                        ida_entry.add_entry(va, va, str(name), 0)
                        api_log.append((va, str(name),dll_name[1:]))

                    make_func_name(ref, name)       
        #idc.MakeByte(image_base)
        ida_bytes.create_data(image_base, FF_BYTE, 1, ida_idaapi.BADADDR)
        
        
        for export_va in lib_exports:
            va = va_from_string(export_va)
            #idc.MakeByte(va)
            ida_bytes.create_data(va, FF_BYTE, 1, ida_idaapi.BADADDR)
            if not idc.get_name(va, ida_name.GN_VISIBLE):
                make_func_name(va,
                    lib_exports[export_va]['name'],
                    (idc.SN_NOCHECK | idc.SN_NOWARN | idc.SN_NOLIST))
        idc.plan_and_wait(dll['image_base'], dll['image_end'])
    

def get_seg(seg):
    'gets all cross-refs to a segment, returns a list'
    seg = va_from_string(seg)
    cross_refs = set()
    # get addresses of the start and end segment
    start = idc.get_segm_start(seg)
    end = idc.get_segm_end(seg)

    # loop through each referenced address
    for ea in idautils.Heads(start, end):
        gen_xrefs = idautils.XrefsTo(ea, 0)
        for _xref in gen_xrefs:
            cross_refs.add(ea)
    return cross_refs


def import_data():
    '''import file to save to buffer'''
    fileName = AskFile(0, "*.*", 'Import File')
    try:
        f = gzip.open(fileName, 'rb')
        temp = f.read()
        f.close()
        return temp
    except Exception as e:
        sys.stdout.write('ERROR: Cannot access file: %s' % e)


def read_json(fn):
    try:
        return json.load(open(fn))
    except Exception as e:
        sys.stdout.write('ERROR: JSON load fail: %s' % e)


def extract_tar_archive(fn, extract_to):
    tar = tarfile.open(fn)
    tar.extractall(extract_to)
    tar.close()
    return extract_to


def extract_gz_archive(fn, extract_to):
    f = gzip.open(fn, 'rb')
    file_content = f.read()
    f.close()
    f = open(extract_to, "wb")
    f.write(file_content)
    f.close()
    return extract_to


def tempdir():
    dn = tempfile.mkdtemp()
    return dn


def load_v2(ida_f, dump_info, temp_dir):
    _unused = ida_f
    idaapi.set_processor_type("p4", ida_idp.SETPROC_LOADER_NON_FATAL|ida_idp.SETPROC_LOADER)

    if dump_info['bitsize'] == IDA_BITSIZE_32:
        seg_mode = 1
    elif dump_info['bitsize'] == IDA_BITSIZE_64:
        seg_mode = 2
    else:
        assert(0)
    entry_point = None

    if 'loaded_libraries' in dump_info:
        for lib in dump_info['loaded_libraries']:
            lib_name = lib['object_description'].split("\\")[-1]
            lib_va = lib['virtual_address']
            lib_exports = lib['exports']
            image_base = va_from_string(lib_va)

            image_end = 0
            for export_va in lib_exports:
                export_va = lib_exports[export_va]['rva'] + image_base
                if export_va > image_end:
                    image_end = export_va
            lib["image_base"] = image_base
            lib["image_end"] = image_end
            idaapi.add_segm(0, image_base, image_end, str(lib_name), "CODE")
            idaapi.set_segm_addressing(idaapi.get_segm_by_name(str(lib_name)), seg_mode)
            idc.set_default_sreg_value(image_base, "es", 0)
            idc.set_default_sreg_value(image_base, "ds", 0)
            idc.set_default_sreg_value(image_base, "cs", 0)

    if 'pe_images' in dump_info:
        for pe_image in dump_info['pe_images']:
            pe_fn = path.join(temp_dir, pe_image['object_uuid'])
            image_base = va_from_string(pe_image['virtual_address'])
            image_end = image_base + os.path.getsize(pe_fn)

            if 'sections' not in pe_image:
                idaapi.add_segm(0, image_base, image_end, str(pe_image['object_uuid']), "UNK")
                idaapi.set_segm_addressing(idaapi.get_segm_by_name(
                    str(pe_image['object_uuid'])), seg_mode)
                idc.set_default_sreg_value(image_base, "es", 0)
                idc.set_default_sreg_value(image_base, "ds", 0)
                idc.set_default_sreg_value(image_base, "cs", 0)
                pe_data = open(pe_fn, "rb").read()
                idaapi.put_bytes(image_base, pe_data)
            else:
                for section in pe_image['sections']:
                    entry_points_rva = section.get('entry_points_rva')
                    if entry_points_rva:
                        for ep in entry_points_rva:
                            va = va_from_string(ep['rva']) + image_base
                            idaapi.add_entry(va, va,
                                str("%s" % ep['description']), 0)

                    characteristics = va_from_string(section['characteristics'])
                    seg_type = "DATA"
                    if characteristics & IMAGE_SCN_MEM_EXECUTE:
                        seg_type = "CODE"

                    section_fn = path.join(temp_dir, section['object_uuid'])

                    va_start = va_from_string(section['virtual_address'])
                    va_end = va_start + os.path.getsize(section_fn)

                    section_name = str(section['name']+ '.' + section['object_uuid'])
                    idaapi.add_segm(0, va_start, va_end, section_name, seg_type)
                    idaapi.set_segm_addressing(idaapi.get_segm_by_name(
                        section_name), seg_mode)
                    idc.set_default_sreg_value(va_start, "es", 0)
                    idc.set_default_sreg_value(va_start, "ds", 0)
                    idc.set_default_sreg_value(va_start, "cs", 0)
                    section_data = open(section_fn, "rb").read()
                    idaapi.put_bytes(va_start, section_data)

            if 'entry_point' in pe_image:
                entry_point = \
                    va_from_string(pe_image['entry_point']) + image_base
                idaapi.add_entry(entry_point, entry_point,
                    str("entry_point.%s" % pe_image['object_uuid']), 1)
                idaapi.auto_make_proc(entry_point)

            idc.plan_and_wait(image_base, image_end)

    memory_blocks = dump_info.get("memory_blocks", None)
    if memory_blocks:
        for block in memory_blocks:
            block_fn = path.join(temp_dir, block['object_uuid'])
            va_start = va_from_string(block['virtual_address'])
            va_end = va_start + os.path.getsize(block_fn)
            access = va_from_string(block['object_data']['access'])
            seg_type = "DATA"
            if ((access & PAGE_EXECUTE) or
               (access & PAGE_EXECUTE_READ) or
               (access & PAGE_EXECUTE_READWRITE) or
               (access & PAGE_EXECUTE_WRITECOPY)):
                seg_type = "CODE"

            idaapi.add_segm(0, va_start, va_end, str(block['object_uuid']), seg_type)
            idaapi.set_segm_addressing(idaapi.get_segm_by_name(
                str(block['object_uuid'])), seg_mode)
            idc.set_default_sreg_value(va_start, "es", 0)
            idc.set_default_sreg_value(va_start, "ds", 0)
            idc.set_default_sreg_value(va_start, "cs", 0)
            block_data = open(block_fn, "rb").read()
            idaapi.put_bytes(va_start, block_data)

            entry_points_rva = block.get('entry_points_rva', None)
            if entry_points_rva:
                for entry_point in entry_points_rva:
                    description = entry_point['description']
                    entry_point = \
                        va_from_string(block['virtual_address']) + \
                        va_from_string(entry_point['rva'])
                    idaapi.add_entry(entry_point, entry_point, str(description), 1)
                    idaapi.auto_make_proc(entry_point)
            idc.plan_and_wait(va_start, va_end)

    if 'loaded_libraries' in dump_info:
        for lib in dump_info['loaded_libraries']:
            print("Analyze area: {} -{}".format(lib['image_base'], lib['image_end']))
            idc.plan_and_wait(lib['image_base'],lib['image_end'])

    print("Loading of the binary data complete! Analyze loaded libraries...")
    analyze_api_refs(dump_info)
    if entry_point:
        ida_kernwin.jumpto(entry_point)


def accept_file(file_stream, absolute_file_path):
    """
    First function required by the IDA loader. It checks the input file format
    and must return a string indicating the name of the file format or 0 for unknown
    input file

    :param file_stream: input file stream
    :type file_stream: filetype obj
    :param absolute_file_path:
        NOTE: in order to maintain backward compatibility with IDA 6, this param
        could assume two different meaning:
        IDA 7.0: string, absolute path of the input file
        IDA 6.*: integer, number of times this function has been called
    :type absolute_file_path:
        IDA 7.0: str
        IDA 6.*: int
    :return: name of the file format or 0 for unknown input file
    :rtype: str or int
    """
    

    if not isinstance(absolute_file_path, str):
        # Called from IDA 6
        if absolute_file_path != 0:
            return 0

    file_stream.seek(0)
    buf = file_stream.read(0x40)
    if buf is None:
        return 0
    if buf.startswith(b"LASTLINE PROCESS DUMP INFO"):
        buf = buf[0x20:]
    else:
        return 0

    version = struct.unpack("I", buf[0:4])[0]
    if version != PROCESS_DUMP_VERSION:
        return 0

    return "Lastline Process Snapshot v.%d" % PROCESS_DUMP_VERSION


def get_snapshot_info(file_name):
    snapshot_info = read_json(file_name)
    return (snapshot_info['snapshot_id'],
            snapshot_info['bitsize'],
            snapshot_info['analysis_reason'],
            file_name)


def create_line_color(start_va, end_va, color = 0xfff2e0):
    """
    adds highlight lines to the selected range
    """
    while start_va <= end_va:
        idaapi.del_item_color(start_va)
        idaapi.set_item_color(start_va, color)
        start_va += idc.get_item_size(start_va)


def load_codehash_info(snapshot_id, ida_bitsize, codehash_dir):
    """
    Load llcodehash info and highlight the covered code

    :param snapshot_id: id of the snapshot
    :type snapshot_id: `int`
    :param ida_bitsize: IDA bitsize. Possible values: IDA_BITSIZE_32, IDA_BITSIZE_64
    :type ida_bitsize: `int`
    :param codehash_dir: directory containing JSON serialized hashes
    :type codehash_dir: `str`
    """
    if ida_bitsize == IDA_BITSIZE_32:
        fn = "hashes_x86.json"
    elif ida_bitsize == IDA_BITSIZE_64:
        fn = "hashes_x86_64.json"
    else:
        return None

    codehash_info = read_json(path.join(codehash_dir,fn))
    if not codehash_info:
        return None

    version = codehash_info.get('version')
    if version > CODEHASH_VERSION:
        return None

    hashes = codehash_info.get('hashes')

    for func_hash, data in hashes.items():
        if snapshot_id not in data['snaps']:
            continue

        hash_blocks = data.get('hash_blocks')
        blocks_str = []
        start_function_addr = None
        if not hash_blocks:
            # Backward compatibility with older llcodehash report format
            for k,v in data.items():
                if k not in ['size', 'snaps']:
                    blocks_str = [k.strip('L'), v.strip('L')]
                    start_va = int(k.strip('L'), 16)
                    end_va = int(v.strip('L'), 16)
                    idc.plan_and_wait(start_va, end_va)
                    create_line_color(start_va, end_va-1)
                    start_function_addr = start_va
                    break
        else:
            start_function_addr = int(data.get('start_addr').strip('L'), 16)
            blocks_str = []
            for start_va_str, end_va_str in hash_blocks.items():
                start_va = int(start_va_str.strip('L'), 16)
                end_va = int(end_va_str.strip('L'), 16)
                idc.plan_and_wait(start_va, end_va)
                create_line_color(start_va, end_va-1)
                blocks_str.append("{}-{}".format(hex(start_va), hex(end_va)))
                if start_function_addr == start_va:
                    continue
                comment = "Block: {}-{} covered by function {}".format(
                    hex(start_va), hex(end_va), hex(start_function_addr))
                idc.set_cmt(start_va, str(comment), 0)

        if not start_function_addr:
            return None

        comment = "Hash: {} - Blocks: \n{}".format(func_hash, "\n".join(blocks_str))

        idc.set_cmt(start_function_addr, str(comment), 0)
        idaapi.auto_make_proc(start_function_addr)

        same_hash_function_blocks = data.get('same_hash_function_blocks', None)
        if same_hash_function_blocks:
            for start_va_str, end_va_str in same_hash_function_blocks.items():
                start_va = int(start_va_str.strip('L'), 16)
                end_va = int(end_va_str.strip('L'), 16)
                idc.plan_and_wait(start_va, end_va)
                create_line_color(start_va, end_va-1)
                comment = "Hash: {} - equal to function {} \n{} - {}".format(
                    func_hash, hex(start_function_addr), hex(start_va), hex(end_va))
                #idc.MakeComm(start_va, str(comment))
                idc.set_cmt(start_va, str(comment), 0)

def load_file(f, neflags, format): # pylint: disable=W0622
    _unused = neflags, format
    file_size = f.seek(0, 2)
    f.seek(0)
    buf = f.read(file_size)
    if buf is None:
        return 0
    if buf.startswith(b"LASTLINE PROCESS DUMP INFO"):
        buf = buf[0x20:]
    else:
        return 0

    version = struct.unpack("I", buf[0:4])[0]
    if version != PROCESS_DUMP_VERSION:
        return retval

    size = struct.unpack("I", buf[4:8])[0]
    if size > len(buf) - 8:
        return 0

    buf = buf[8:size + 8]

    tmp_dir = tempdir()
    file_path_tar = path.join(tmp_dir, "process_snapshot.tar")
    file_path_tar_gz = file_path_tar + ".gz"
    f_gz = open(file_path_tar_gz, "wb")
    f_gz.write(buf)
    f_gz.close()

    extract_gz_archive(file_path_tar_gz, file_path_tar)
    extract_tar_archive(file_path_tar, tmp_dir)

    if idc.__EA64__:
        ida_bitsize = IDA_BITSIZE_64
    else:
        ida_bitsize = IDA_BITSIZE_32

    snapshots_list = []
    for (dirpath, _dirnames, filenames) in walk(tmp_dir):
        for name in filenames:
            if name.startswith("snapshot") and name.endswith(".json"):
                snapshots_list.append(
                    get_snapshot_info(path.join(dirpath, name)))
        break

    v = idaapi.simplecustviewer_t()
    if v.Create("Process Snapshots"):

        c = "%13s %13s %23s" % ("Snapshot Id", "Bitsize", "Analysis Reason")
        comment = idaapi.COLSTR(c, idaapi.SCOLOR_BINPREF)
        v.AddLine(comment)

        av_snapshots = {}
        unav_snapshots = {}
        def getKey(item):
            return item[0]
        snapshots_list = sorted(snapshots_list, key=getKey)

        for snapshot in snapshots_list:
            line = str("%12s  %12s          %s" % (snapshot[0], snapshot[1], snapshot[2]))
            if ida_bitsize == snapshot[1]:
                v.AddLine(idaapi.COLSTR("%-80s" % line, idaapi.SCOLOR_REG))
                av_snapshots[snapshot[0]] = snapshot
            else:
                line = "%s - use IDA Pro %s bit to open this snapshot" % (line, snapshot[1])
                v.AddLine(idaapi.COLSTR("%-100s" % line, idaapi.SCOLOR_ERROR))
                
                unav_snapshots[snapshot[0]] = snapshot
                unav_bitsize = snapshot[1]

        v.Show()
        #get absolute path of the snapshot
        input_absolute_path = idc.get_input_file_path()
        
        sample_hash = os.path.split(input_absolute_path)[-1]
        if av_snapshots:
            message = {"method": "push", "data": {"hash": sample_hash, "data": {}}}
            # This is to save the send the output to mongodb server
            
            toSend = json.dumps(message)
            #toSend is a json dictionary. It can either be saved as a file or be sent to a DB.
            # Current implementation assumes you are running a mongodb server on localhost
            p = producer.Producer()
            for snapshot_id in av_snapshots:
                snapshot_info = read_json(path.join(tmp_dir,av_snapshots[snapshot_id][3]))
                load_v2(f, snapshot_info, tmp_dir)
                load_codehash_info(snapshot_id, ida_bitsize, tmp_dir)
                ret, exception = cfg_extractor(snapshot_id, sample_hash)
            p.killConnection()
            # kill the connection to the server
            #return 1       
        elif unav_snapshots:
            ida_pro.qexit(0)
            idc.Fatal("Use IDA Pro %s bit to open these process snapshots"\
            % unav_bitsize)
        else:
            ida_pro.qexit(0)
            idc.Fatal("The process snapshots file is invalid")
    ida_pro.qexit(0)
    return 0

