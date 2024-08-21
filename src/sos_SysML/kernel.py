#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import json
from collections.abc import Sequence

import numpy
import pandas
from sos.utils import env, short_repr

SysML_init_statement = r'''

# require 'DASysML' #todo

# require 'nmatrix'

def __SysML_py_repr(obj)
  if obj.is_a? Integer
    return obj.inspect
  elsif obj.is_a? String
    return obj.inspect
  elsif obj.is_a? TrueClass
    return "True"
  elsif obj.is_a? FalseClass
    return "False"
  elsif obj.is_a? Float
    return obj.inspect
  elsif obj.nil?
    return "None"
  elsif obj.is_a? Set
    return "{" + (obj.map { |indivial_var| __SysML_py_repr(indivial_var) } ).join(",") + "}"
  elsif obj.is_a? Range
    return "range(" + obj.min().inspect + "," + (obj.max()+1).inspect + ")"
  elsif obj.is_a? Array
    return '[' + (obj.map { |indivial_var| __SysML_py_repr(indivial_var) } ).join(",") + ']'
  elsif obj.is_a? Hash
    _beginning_result_string_hash_from_SysML = "{"
    _context_result_string_hash_from_SysML = (obj.keys.map do |x|
                                              if obj[x].is_a? Array then
                                                  "\"" + x.to_s + "\":" + (obj[x].to_a.map { |y|  eval(__SysML_py_repr(y)) }).to_s
                                              else
                                                  "\"" + x.to_s + "\":" + (__SysML_py_repr(obj[x])).to_s
                                              end
                                            end).join(",") + "}"
    _result_string_hash_from_SysML = _beginning_result_string_hash_from_SysML + _context_result_string_hash_from_SysML
    return _result_string_hash_from_SysML
#  elsif obj.is_a? SysML::DataFrame
#    _beginning_result_string_dataframe_from_SysML = "pandas.DataFrame(" + "{"
#    _context_result_string_dataframe_from_SysML = (obj.vectors.to_a.map { |x| "\"" + x.to_s + "\":" + (obj[x].to_a.map { |y|  eval(__SysML_py_repr(y)) }).to_s } ).join(",")
#    _indexing_result_string_dataframe_from_SysML = "}," + "index=" + obj.index.to_a.to_s + ")"
#    _result_string_dataframe_from_SysML = _beginning_result_string_dataframe_from_SysML + _context_result_string_dataframe_from_SysML + _indexing_result_string_dataframe_from_SysML
#    return _result_string_dataframe_from_SysML
# elsif obj.is_a SysML::element 
# elsif obj.is_a SysML::instance 
# elsif obj.is_a SysML::definition
# elsif obj.is_a SysML::metadef
  elsif obj.is_a? Complex
    return "complex(" + obj.real.inspect + "," + obj.imaginary.inspect + ")"
  else
    return "'Untransferrable variable'"
  end
end
'''

#
#  support for %get
#
#  Converting a Python object to a JSON format to be loaded by SysML
#


class sos_SysML:
    supported_kernels = {'SysML': ['SysML']}
    background_color = '#e8c2be'
    options = {}
    cd_command = 'Dir.chdir {dir!r}'

    def __init__(self, sos_kernel, kernel_name='SysML'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = SysML_init_statement

    def _SysML_repr(self, obj):
        if isinstance(obj, bool):
            return 'true' if obj else 'false'
        if isinstance(obj, float) and numpy.isnan(obj):
            return "Float::NAN"
        if isinstance(obj, (int, float)):
            return repr(obj)
        if isinstance(obj, str):
            return '%(' + obj + ')'
        if isinstance(obj, complex):
            return 'Complex(' + str(obj.real) + ',' + str(obj.imag) + ')'
        if isinstance(obj, range):
            return '(' + repr(min(obj)) + '...' + repr(max(obj)) + ')'
        if isinstance(obj, Sequence):
            if len(obj) == 0:
                return '[]'
            return '[' + ','.join(self._SysML_repr(x) for x in obj) + ']'
        if obj is None:
            return 'nil'
        if isinstance(obj, dict):
            return '{' + ','.join(f'"{x}" => {self._SysML_repr(y)}' for x, y in obj.items()) + '}'
        if isinstance(obj, set):
            return 'Set[' + ','.join(self._SysML_repr(x) for x in obj) + ']'
        if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, numpy.float64)):
            return repr(obj)
        if isinstance(obj, numpy.matrixlib.defmatrix.matrix):
            return 'N' + repr(obj.tolist())
        if isinstance(obj, numpy.ndarray):
            return repr(obj.tolist())
        if isinstance(obj, pandas.DataFrame):
            _beginning_result_string_dataframe_to_SysML = "Sysml::DataFrame.new({"
            _context_string_dataframe_to_SysML = str(['"'
                                                    + str(x).replace("'", '"')
                                                    + '"'
                                                    + "=>"
                                                    + "["
                                                    + str(
                                                        ",".join(
                                                            list(
                                                            map(
                                                                lambda y: self._SysML_repr(y),
                                                                    obj[x].tolist()
                                                            )
                                                        )
                                                        )
                                                    ).replace("'", '"') + "]"
                                                    for x in obj.keys().tolist()])[2:-2].replace("\', \'", ", ") + "},"
            _indexing_result_string_dataframe_to_SysML = "index:" + str(obj.index.values.tolist()).replace("'", '"') + ")"
            _result_string_dataframe_to_SysML = _beginning_result_string_dataframe_to_SysML + _context_string_dataframe_to_SysML + _indexing_result_string_dataframe_to_SysML
            return _result_string_dataframe_to_SysML
        if isinstance(obj, pandas.Series):
            dat=list(obj.values)
            ind=list(obj.index.values)
            ans="{" + ",".join([repr(x) + "=>" + repr(y) for x, y in zip(ind, dat)]) + "}"
            return ans
        return repr(f'Unsupported datatype {short_repr(obj)}')

    async def get_vars(self, names, as_var=None):
        for name in names:
            newname = as_var if as_var else name
            SysML_repr = self._SysML_repr(env.sos_dict[name])
            await self.sos_kernel.run_cell(f'{newname} = {SysML_repr}', True, False,
                                     on_error=f'Failed to put variable {name} to SysML')

    def put_vars(self, items, to_kernel=None, as_var=None):
        # first let us get all variables with names starting with sos
        try:
            response = self.sos_kernel.get_response('print local_variables', ('stream',), name=('stdout',))[0][1]
            all_vars = response['text']
            items += [x for x in all_vars[1:-1].split(", ") if x.startswith(":sos")]
        except:
            # if there is no variable with name sos, the command will not produce any output
            pass
        res = {}
        for item in items:
            py_repr = f'print(__SysML_py_repr({item}))'
            response = self.sos_kernel.get_response(py_repr, ('stream',), name=('stdout',))[0][1]
            expr = response['text']
            self.sos_kernel.warn(repr(expr))

            try:
                # evaluate as raw string to correctly handle \\ etc
                res[as_var if as_var else item] = eval(expr)
            except Exception as e:
                self.sos_kernel.warn(f'Failed to evaluate {expr!r}: {e}')
                return None
        return res

    def sessioninfo(self):
        response = self.sos_kernel.get_response(r'SYSML_VERSION', ('stream',), name=('stdout',))
        return response['text']
