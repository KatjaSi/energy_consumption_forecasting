═║"
ђ¤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.11.02v2.11.0-rc2-15-g6290819256d8Ею 
Ў
 Adam/lstm_15/lstm_cell_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*1
shared_name" Adam/lstm_15/lstm_cell_19/bias/v
њ
4Adam/lstm_15/lstm_cell_19/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_15/lstm_cell_19/bias/v*
_output_shapes	
:љ*
dtype0
х
,Adam/lstm_15/lstm_cell_19/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dљ*=
shared_name.,Adam/lstm_15/lstm_cell_19/recurrent_kernel/v
«
@Adam/lstm_15/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_15/lstm_cell_19/recurrent_kernel/v*
_output_shapes
:	dљ*
dtype0
б
"Adam/lstm_15/lstm_cell_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*3
shared_name$"Adam/lstm_15/lstm_cell_19/kernel/v
Џ
6Adam/lstm_15/lstm_cell_19/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_15/lstm_cell_19/kernel/v* 
_output_shapes
:
цљ*
dtype0
Ў
 Adam/lstm_14/lstm_cell_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*1
shared_name" Adam/lstm_14/lstm_cell_18/bias/v
њ
4Adam/lstm_14/lstm_cell_18/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_18/bias/v*
_output_shapes	
:љ*
dtype0
Х
,Adam/lstm_14/lstm_cell_18/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*=
shared_name.,Adam/lstm_14/lstm_cell_18/recurrent_kernel/v
»
@Adam/lstm_14/lstm_cell_18/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_18/recurrent_kernel/v* 
_output_shapes
:
цљ*
dtype0
А
"Adam/lstm_14/lstm_cell_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*3
shared_name$"Adam/lstm_14/lstm_cell_18/kernel/v
џ
6Adam/lstm_14/lstm_cell_18/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_18/kernel/v*
_output_shapes
:	љ*
dtype0
ђ
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/v
Ђ
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_14/kernel/v
Ђ
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:d*
dtype0
Ў
 Adam/lstm_15/lstm_cell_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*1
shared_name" Adam/lstm_15/lstm_cell_19/bias/m
њ
4Adam/lstm_15/lstm_cell_19/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_15/lstm_cell_19/bias/m*
_output_shapes	
:љ*
dtype0
х
,Adam/lstm_15/lstm_cell_19/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dљ*=
shared_name.,Adam/lstm_15/lstm_cell_19/recurrent_kernel/m
«
@Adam/lstm_15/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_15/lstm_cell_19/recurrent_kernel/m*
_output_shapes
:	dљ*
dtype0
б
"Adam/lstm_15/lstm_cell_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*3
shared_name$"Adam/lstm_15/lstm_cell_19/kernel/m
Џ
6Adam/lstm_15/lstm_cell_19/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_15/lstm_cell_19/kernel/m* 
_output_shapes
:
цљ*
dtype0
Ў
 Adam/lstm_14/lstm_cell_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*1
shared_name" Adam/lstm_14/lstm_cell_18/bias/m
њ
4Adam/lstm_14/lstm_cell_18/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_18/bias/m*
_output_shapes	
:љ*
dtype0
Х
,Adam/lstm_14/lstm_cell_18/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*=
shared_name.,Adam/lstm_14/lstm_cell_18/recurrent_kernel/m
»
@Adam/lstm_14/lstm_cell_18/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_18/recurrent_kernel/m* 
_output_shapes
:
цљ*
dtype0
А
"Adam/lstm_14/lstm_cell_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*3
shared_name$"Adam/lstm_14/lstm_cell_18/kernel/m
џ
6Adam/lstm_14/lstm_cell_18/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_18/kernel/m*
_output_shapes
:	љ*
dtype0
ђ
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/m
Ђ
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_14/kernel/m
Ђ
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
І
lstm_15/lstm_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ**
shared_namelstm_15/lstm_cell_19/bias
ё
-lstm_15/lstm_cell_19/bias/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_19/bias*
_output_shapes	
:љ*
dtype0
Д
%lstm_15/lstm_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dљ*6
shared_name'%lstm_15/lstm_cell_19/recurrent_kernel
а
9lstm_15/lstm_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_15/lstm_cell_19/recurrent_kernel*
_output_shapes
:	dљ*
dtype0
ћ
lstm_15/lstm_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*,
shared_namelstm_15/lstm_cell_19/kernel
Ї
/lstm_15/lstm_cell_19/kernel/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_19/kernel* 
_output_shapes
:
цљ*
dtype0
І
lstm_14/lstm_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ**
shared_namelstm_14/lstm_cell_18/bias
ё
-lstm_14/lstm_cell_18/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_18/bias*
_output_shapes	
:љ*
dtype0
е
%lstm_14/lstm_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
цљ*6
shared_name'%lstm_14/lstm_cell_18/recurrent_kernel
А
9lstm_14/lstm_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_18/recurrent_kernel* 
_output_shapes
:
цљ*
dtype0
Њ
lstm_14/lstm_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*,
shared_namelstm_14/lstm_cell_18/kernel
ї
/lstm_14/lstm_cell_18/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_18/kernel*
_output_shapes
:	љ*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:d*
dtype0
ѕ
serving_default_lstm_14_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
╦
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_14_inputlstm_14/lstm_cell_18/kernel%lstm_14/lstm_cell_18/recurrent_kernellstm_14/lstm_cell_18/biaslstm_15/lstm_cell_19/kernel%lstm_15/lstm_cell_19/recurrent_kernellstm_15/lstm_cell_19/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference_signature_wrapper_1467784

NoOpNoOp
љK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╦J
value┴JBЙJ BиJ
У
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
д
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
д
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
J
00
11
22
33
44
55
&6
'7
.8
/9*
J
00
11
22
33
44
55
&6
'7
.8
/9*
* 
░
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
;trace_0
<trace_1
=trace_2
>trace_3* 
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
* 
ї
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_rate&mџ'mЏ.mю/mЮ0mъ1mЪ2mа3mА4mб5mБ&vц'vЦ.vд/vД0vе1vЕ2vф3vФ4vг5vГ*

Hserving_default* 

00
11
22*

00
11
22*
* 
Ъ

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
* 
с
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator
^
state_size

0kernel
1recurrent_kernel
2bias*
* 

30
41
52*

30
41
52*
* 
Ъ

_states
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
etrace_0
ftrace_1
gtrace_2
htrace_3* 
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
* 
с
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator
t
state_size

3kernel
4recurrent_kernel
5bias*
* 

&0
'1*

&0
'1*
* 
Њ
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

ztrace_0* 

{trace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
ћ
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
ђlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Ђtrace_0* 

ѓtrace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_14/lstm_cell_18/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_14/lstm_cell_18/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_14/lstm_cell_18/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_15/lstm_cell_19/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_15/lstm_cell_19/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_15/lstm_cell_19/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

Ѓ0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

00
11
22*

00
11
22*
* 
ў
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Ѕtrace_0
іtrace_1* 

Іtrace_0
їtrace_1* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30
41
52*

30
41
52*
* 
ў
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

њtrace_0
Њtrace_1* 

ћtrace_0
Ћtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ќ	variables
Ќ	keras_api

ўtotal

Ўcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ў0
Ў1*

ќ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_18/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_18/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_14/lstm_cell_18/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_15/lstm_cell_19/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_15/lstm_cell_19/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_15/lstm_cell_19/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_18/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_18/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_14/lstm_cell_18/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_15/lstm_cell_19/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_15/lstm_cell_19/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_15/lstm_cell_19/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
■
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp/lstm_14/lstm_cell_18/kernel/Read/ReadVariableOp9lstm_14/lstm_cell_18/recurrent_kernel/Read/ReadVariableOp-lstm_14/lstm_cell_18/bias/Read/ReadVariableOp/lstm_15/lstm_cell_19/kernel/Read/ReadVariableOp9lstm_15/lstm_cell_19/recurrent_kernel/Read/ReadVariableOp-lstm_15/lstm_cell_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_18/kernel/m/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_18/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_18/bias/m/Read/ReadVariableOp6Adam/lstm_15/lstm_cell_19/kernel/m/Read/ReadVariableOp@Adam/lstm_15/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_15/lstm_cell_19/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_18/kernel/v/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_18/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_18/bias/v/Read/ReadVariableOp6Adam/lstm_15/lstm_cell_19/kernel/v/Read/ReadVariableOp@Adam/lstm_15/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_15/lstm_cell_19/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__traced_save_1470037
Ћ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biasdense_15/kerneldense_15/biaslstm_14/lstm_cell_18/kernel%lstm_14/lstm_cell_18/recurrent_kernellstm_14/lstm_cell_18/biaslstm_15/lstm_cell_19/kernel%lstm_15/lstm_cell_19/recurrent_kernellstm_15/lstm_cell_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/m"Adam/lstm_14/lstm_cell_18/kernel/m,Adam/lstm_14/lstm_cell_18/recurrent_kernel/m Adam/lstm_14/lstm_cell_18/bias/m"Adam/lstm_15/lstm_cell_19/kernel/m,Adam/lstm_15/lstm_cell_19/recurrent_kernel/m Adam/lstm_15/lstm_cell_19/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v"Adam/lstm_14/lstm_cell_18/kernel/v,Adam/lstm_14/lstm_cell_18/recurrent_kernel/v Adam/lstm_14/lstm_cell_18/bias/v"Adam/lstm_15/lstm_cell_19/kernel/v,Adam/lstm_15/lstm_cell_19/recurrent_kernel/v Adam/lstm_15/lstm_cell_19/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference__traced_restore_1470158┼Я
»8
і
D__inference_lstm_14_layer_call_and_return_conditional_losses_1466511

inputs'
lstm_cell_18_1466429:	љ(
lstm_cell_18_1466431:
цљ#
lstm_cell_18_1466433:	љ
identityѕб$lstm_cell_18/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЧ
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_1466429lstm_cell_18_1466431lstm_cell_18_1466433*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466383n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : └
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_1466429lstm_cell_18_1466431lstm_cell_18_1466433*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1466442*
condR
while_cond_1466441*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  цu
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ћ

§
%__inference_signature_wrapper_1467784
lstm_14_input
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
	unknown_2:
цљ
	unknown_3:	dљ
	unknown_4:	љ
	unknown_5:d
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__wrapped_model_1466170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
 J
Ъ
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468615
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1468531*
condR
while_cond_1468530*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
▄A
М

lstm_14_while_body_1468190,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0:	љQ
=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљK
<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_18_matmul_readvariableop_resource:	љO
;lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource:
цљI
:lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpб0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpб2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpљ
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Г
0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0м
!lstm_14/while/lstm_cell_18/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ▓
2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0╣
#lstm_14/while/lstm_cell_18/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љХ
lstm_14/while/lstm_cell_18/addAddV2+lstm_14/while/lstm_cell_18/MatMul:product:0-lstm_14/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љФ
1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0┐
"lstm_14/while/lstm_cell_18/BiasAddBiasAdd"lstm_14/while/lstm_cell_18/add:z:09lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љl
*lstm_14/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :І
 lstm_14/while/lstm_cell_18/splitSplit3lstm_14/while/lstm_cell_18/split/split_dim:output:0+lstm_14/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitІ
"lstm_14/while/lstm_cell_18/SigmoidSigmoid)lstm_14/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цЇ
$lstm_14/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЪ
lstm_14/while/lstm_cell_18/mulMul(lstm_14/while/lstm_cell_18/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         цЁ
lstm_14/while/lstm_cell_18/ReluRelu)lstm_14/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         ц▒
 lstm_14/while/lstm_cell_18/mul_1Mul&lstm_14/while/lstm_cell_18/Sigmoid:y:0-lstm_14/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цд
 lstm_14/while/lstm_cell_18/add_1AddV2"lstm_14/while/lstm_cell_18/mul:z:0$lstm_14/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цЇ
$lstm_14/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цѓ
!lstm_14/while/lstm_cell_18/Relu_1Relu$lstm_14/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цх
 lstm_14/while/lstm_cell_18/mul_2Mul(lstm_14/while/lstm_cell_18/Sigmoid_2:y:0/lstm_14/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цт
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: і
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: q
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: ъ
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: њ
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_18/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         цњ
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_18/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ц­
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_18_matmul_readvariableop_resource;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0"╚
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2f
1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
╚	
Ш
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю

Ш
E__inference_dense_14_layer_call_and_return_conditional_losses_1469688

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ЬB
М

lstm_15_while_body_1468033,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0:
цљP
=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљK
<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_19_matmul_readvariableop_resource:
цљN
;lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource:	dљI
:lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpб0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpб2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpљ
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ¤
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0«
0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0м
!lstm_15/while/lstm_cell_19/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ▒
2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0╣
#lstm_15/while/lstm_cell_19/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љХ
lstm_15/while/lstm_cell_19/addAddV2+lstm_15/while/lstm_cell_19/MatMul:product:0-lstm_15/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љФ
1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0┐
"lstm_15/while/lstm_cell_19/BiasAddBiasAdd"lstm_15/while/lstm_cell_19/add:z:09lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љl
*lstm_15/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_15/while/lstm_cell_19/splitSplit3lstm_15/while/lstm_cell_19/split/split_dim:output:0+lstm_15/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitі
"lstm_15/while/lstm_cell_19/SigmoidSigmoid)lstm_15/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dї
$lstm_15/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dъ
lstm_15/while/lstm_cell_19/mulMul(lstm_15/while/lstm_cell_19/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:         dё
lstm_15/while/lstm_cell_19/ReluRelu)lstm_15/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         d░
 lstm_15/while/lstm_cell_19/mul_1Mul&lstm_15/while/lstm_cell_19/Sigmoid:y:0-lstm_15/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЦ
 lstm_15/while/lstm_cell_19/add_1AddV2"lstm_15/while/lstm_cell_19/mul:z:0$lstm_15/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dї
$lstm_15/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dЂ
!lstm_15/while/lstm_cell_19/Relu_1Relu$lstm_15/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         d┤
 lstm_15/while/lstm_cell_19/mul_2Mul(lstm_15/while/lstm_cell_19/Sigmoid_2:y:0/lstm_15/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dz
8lstm_15/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1Alstm_15/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_15/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: і
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: q
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: ъ
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: Љ
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_19/mul_2:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:         dЉ
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_19/add_1:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:         d­
lstm_15/while/NoOpNoOp2^lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"z
:lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_15_while_lstm_cell_19_matmul_readvariableop_resource;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0"╚
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2f
1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
Й
╚
while_cond_1468816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1468816___redundant_placeholder05
1while_while_cond_1468816___redundant_placeholder15
1while_while_cond_1468816___redundant_placeholder25
1while_while_cond_1468816___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
Й
╚
while_cond_1466250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1466250___redundant_placeholder05
1while_while_cond_1466250___redundant_placeholder15
1while_while_cond_1466250___redundant_placeholder25
1while_while_cond_1466250___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
┼

є
.__inference_sequential_7_layer_call_fn_1467695
lstm_14_input
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
	unknown_2:
цљ
	unknown_3:	dљ
	unknown_4:	љ
	unknown_5:d
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
Ј9
і
D__inference_lstm_15_layer_call_and_return_conditional_losses_1466865

inputs(
lstm_cell_19_1466781:
цљ'
lstm_cell_19_1466783:	dљ#
lstm_cell_19_1466785:	љ
identityѕб$lstm_cell_19/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskщ
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_1466781lstm_cell_19_1466783lstm_cell_19_1466785*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466735n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_1466781lstm_cell_19_1466783lstm_cell_19_1466785*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1466795*
condR
while_cond_1466794*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         du
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ц
 
_user_specified_nameinputs
Б8
М
while_body_1467499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
Э
Э
.__inference_lstm_cell_18_layer_call_fn_1469724

inputs
states_0
states_1
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identity

identity_1

identity_2ѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466237p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         цr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         цr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_1
Щб
у
#__inference__traced_restore_1470158
file_prefix2
 assignvariableop_dense_14_kernel:d.
 assignvariableop_1_dense_14_bias:4
"assignvariableop_2_dense_15_kernel:.
 assignvariableop_3_dense_15_bias:A
.assignvariableop_4_lstm_14_lstm_cell_18_kernel:	љL
8assignvariableop_5_lstm_14_lstm_cell_18_recurrent_kernel:
цљ;
,assignvariableop_6_lstm_14_lstm_cell_18_bias:	љB
.assignvariableop_7_lstm_15_lstm_cell_19_kernel:
цљK
8assignvariableop_8_lstm_15_lstm_cell_19_recurrent_kernel:	dљ;
,assignvariableop_9_lstm_15_lstm_cell_19_bias:	љ'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_14_kernel_m:d6
(assignvariableop_18_adam_dense_14_bias_m:<
*assignvariableop_19_adam_dense_15_kernel_m:6
(assignvariableop_20_adam_dense_15_bias_m:I
6assignvariableop_21_adam_lstm_14_lstm_cell_18_kernel_m:	љT
@assignvariableop_22_adam_lstm_14_lstm_cell_18_recurrent_kernel_m:
цљC
4assignvariableop_23_adam_lstm_14_lstm_cell_18_bias_m:	љJ
6assignvariableop_24_adam_lstm_15_lstm_cell_19_kernel_m:
цљS
@assignvariableop_25_adam_lstm_15_lstm_cell_19_recurrent_kernel_m:	dљC
4assignvariableop_26_adam_lstm_15_lstm_cell_19_bias_m:	љ<
*assignvariableop_27_adam_dense_14_kernel_v:d6
(assignvariableop_28_adam_dense_14_bias_v:<
*assignvariableop_29_adam_dense_15_kernel_v:6
(assignvariableop_30_adam_dense_15_bias_v:I
6assignvariableop_31_adam_lstm_14_lstm_cell_18_kernel_v:	љT
@assignvariableop_32_adam_lstm_14_lstm_cell_18_recurrent_kernel_v:
цљC
4assignvariableop_33_adam_lstm_14_lstm_cell_18_bias_v:	љJ
6assignvariableop_34_adam_lstm_15_lstm_cell_19_kernel_v:
цљS
@assignvariableop_35_adam_lstm_15_lstm_cell_19_recurrent_kernel_v:	dљC
4assignvariableop_36_adam_lstm_15_lstm_cell_19_bias_v:	љ
identity_38ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ў
valueјBІ&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ▀
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*«
_output_shapesЏ
ў::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_15_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_15_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_14_lstm_cell_18_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_5AssignVariableOp8assignvariableop_5_lstm_14_lstm_cell_18_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_6AssignVariableOp,assignvariableop_6_lstm_14_lstm_cell_18_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_15_lstm_cell_19_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_15_lstm_cell_19_recurrent_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_15_lstm_cell_19_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_14_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_14_bias_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_15_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_15_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_lstm_14_lstm_cell_18_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_lstm_14_lstm_cell_18_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_14_lstm_cell_18_bias_mIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_15_lstm_cell_19_kernel_mIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_15_lstm_cell_19_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_15_lstm_cell_19_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_14_kernel_vIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_14_bias_vIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_15_kernel_vIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_15_bias_vIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_lstm_14_lstm_cell_18_kernel_vIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_lstm_14_lstm_cell_18_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_lstm_14_lstm_cell_18_bias_vIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_lstm_15_lstm_cell_19_kernel_vIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_35AssignVariableOp@assignvariableop_35_adam_lstm_15_lstm_cell_19_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_lstm_15_lstm_cell_19_bias_vIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 §
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: Ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
─
Ќ
*__inference_dense_14_layer_call_fn_1469677

inputs
unknown:d
	unknown_0:
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ч
и
)__inference_lstm_15_layer_call_fn_1469088

inputs
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
рх
И	
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468428

inputsF
3lstm_14_lstm_cell_18_matmul_readvariableop_resource:	љI
5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource:
цљC
4lstm_14_lstm_cell_18_biasadd_readvariableop_resource:	љG
3lstm_15_lstm_cell_19_matmul_readvariableop_resource:
цљH
5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource:	dљC
4lstm_15_lstm_cell_19_biasadd_readvariableop_resource:	љ9
'dense_14_matmul_readvariableop_resource:d6
(dense_14_biasadd_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpб*lstm_14/lstm_cell_18/MatMul/ReadVariableOpб,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpбlstm_14/whileб+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpб*lstm_15/lstm_cell_19/MatMul/ReadVariableOpб,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpбlstm_15/whileC
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цІ
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         ц[
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цЈ
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         цk
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:g
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЪ
*lstm_14/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0«
lstm_14/lstm_cell_18/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0е
lstm_14/lstm_cell_18/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
lstm_14/lstm_cell_18/addAddV2%lstm_14/lstm_cell_18/MatMul:product:0'lstm_14/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЮ
+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Г
lstm_14/lstm_cell_18/BiasAddBiasAddlstm_14/lstm_cell_18/add:z:03lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љf
$lstm_14/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :щ
lstm_14/lstm_cell_18/splitSplit-lstm_14/lstm_cell_18/split/split_dim:output:0%lstm_14/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split
lstm_14/lstm_cell_18/SigmoidSigmoid#lstm_14/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цЂ
lstm_14/lstm_cell_18/Sigmoid_1Sigmoid#lstm_14/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цљ
lstm_14/lstm_cell_18/mulMul"lstm_14/lstm_cell_18/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         цy
lstm_14/lstm_cell_18/ReluRelu#lstm_14/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЪ
lstm_14/lstm_cell_18/mul_1Mul lstm_14/lstm_cell_18/Sigmoid:y:0'lstm_14/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цћ
lstm_14/lstm_cell_18/add_1AddV2lstm_14/lstm_cell_18/mul:z:0lstm_14/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цЂ
lstm_14/lstm_cell_18/Sigmoid_2Sigmoid#lstm_14/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цv
lstm_14/lstm_cell_18/Relu_1Relulstm_14/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цБ
lstm_14/lstm_cell_18/mul_2Mul"lstm_14/lstm_cell_18/Sigmoid_2:y:0)lstm_14/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цv
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   л
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_18_matmul_readvariableop_resource5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource4lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_14_while_body_1468190*&
condR
lstm_14_while_cond_1468189*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ѕ
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   █
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0p
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskm
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         цc
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_15/ShapeShapelstm_14/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dІ
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:         dZ
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЈ
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:         dk
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
lstm_15/transpose	Transposelstm_14/transpose_1:y:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         цT
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:g
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Э
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskа
*lstm_15/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0«
lstm_15/lstm_cell_19/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љБ
,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0е
lstm_15/lstm_cell_19/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
lstm_15/lstm_cell_19/addAddV2%lstm_15/lstm_cell_19/MatMul:product:0'lstm_15/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЮ
+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Г
lstm_15/lstm_cell_19/BiasAddBiasAddlstm_15/lstm_cell_19/add:z:03lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љf
$lstm_15/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_15/lstm_cell_19/splitSplit-lstm_15/lstm_cell_19/split/split_dim:output:0%lstm_15/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_split~
lstm_15/lstm_cell_19/SigmoidSigmoid#lstm_15/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dђ
lstm_15/lstm_cell_19/Sigmoid_1Sigmoid#lstm_15/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dЈ
lstm_15/lstm_cell_19/mulMul"lstm_15/lstm_cell_19/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:         dx
lstm_15/lstm_cell_19/ReluRelu#lstm_15/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dъ
lstm_15/lstm_cell_19/mul_1Mul lstm_15/lstm_cell_19/Sigmoid:y:0'lstm_15/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЊ
lstm_15/lstm_cell_19/add_1AddV2lstm_15/lstm_cell_19/mul:z:0lstm_15/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dђ
lstm_15/lstm_cell_19/Sigmoid_2Sigmoid#lstm_15/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         du
lstm_15/lstm_cell_19/Relu_1Relulstm_15/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dб
lstm_15/lstm_cell_19/mul_2Mul"lstm_15/lstm_cell_19/Sigmoid_2:y:0)lstm_15/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dv
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   f
$lstm_15/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0-lstm_15/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : З
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_19_matmul_readvariableop_resource5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource4lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_15_while_body_1468330*&
condR
lstm_15_while_cond_1468329*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ѕ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   Ь
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsp
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maskm
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:         dc
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    є
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ћ
dense_14/MatMulMatMul lstm_15/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ђ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp,^lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_18/MatMul/ReadVariableOp-^lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_19/MatMul/ReadVariableOp-^lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2Z
+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_18/MatMul/ReadVariableOp*lstm_14/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while2Z
+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_15/lstm_cell_19/MatMul/ReadVariableOp*lstm_15/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▓K
Ю
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469523

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1469438*
condR
while_cond_1469437*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
Б8
М
while_body_1468531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
В
є
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466237

inputs

states
states_11
matmul_readvariableop_resource:	љ4
 matmul_1_readvariableop_resource:
цљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         цW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         цV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         цO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ц`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         цU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         цW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         цL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         цd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         цY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         цЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ц
 
_user_specified_namestates:PL
(
_output_shapes
:         ц
 
_user_specified_namestates
Г9
М
while_body_1469438
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
НK
Ъ
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469233
inputs_0?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1469148*
condR
while_cond_1469147*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ц
"
_user_specified_name
inputs_0
║
╚
while_cond_1466794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1466794___redundant_placeholder05
1while_while_cond_1466794___redundant_placeholder15
1while_while_cond_1466794___redundant_placeholder25
1while_while_cond_1466794___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
║
╚
while_cond_1469582
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1469582___redundant_placeholder05
1while_while_cond_1469582___redundant_placeholder15
1while_while_cond_1469582___redundant_placeholder25
1while_while_cond_1469582___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
ё
и
)__inference_lstm_14_layer_call_fn_1468472

inputs
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467583t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 J
Ъ
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468758
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1468674*
condR
while_cond_1468673*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
­
Э
.__inference_lstm_cell_19_layer_call_fn_1469822

inputs
states_0
states_1
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states_0:QM
'
_output_shapes
:         d
"
_user_specified_name
states_1
ё
и
)__inference_lstm_14_layer_call_fn_1468461

inputs
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467023t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Б8
М
while_body_1466939
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
І
В
'sequential_7_lstm_14_while_cond_1465931F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3H
Dsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1_
[sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_1465931___redundant_placeholder0_
[sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_1465931___redundant_placeholder1_
[sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_1465931___redundant_placeholder2_
[sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_1465931___redundant_placeholder3'
#sequential_7_lstm_14_while_identity
Х
sequential_7/lstm_14/while/LessLess&sequential_7_lstm_14_while_placeholderDsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_14/while/IdentityIdentity#sequential_7/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_14_while_identity,sequential_7/lstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
Џ

У
lstm_15_while_cond_1468032,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1E
Alstm_15_while_lstm_15_while_cond_1468032___redundant_placeholder0E
Alstm_15_while_lstm_15_while_cond_1468032___redundant_placeholder1E
Alstm_15_while_lstm_15_while_cond_1468032___redundant_placeholder2E
Alstm_15_while_lstm_15_while_cond_1468032___redundant_placeholder3
lstm_15_while_identity
ѓ
lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: [
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Р
ѕ
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469871

inputs
states_0
states_12
matmul_readvariableop_resource:
цљ3
 matmul_1_readvariableop_resource:	dљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         dЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states_0:QM
'
_output_shapes
:         d
"
_user_specified_name
states_1
з┘
Х
"__inference__wrapped_model_1466170
lstm_14_inputS
@sequential_7_lstm_14_lstm_cell_18_matmul_readvariableop_resource:	љV
Bsequential_7_lstm_14_lstm_cell_18_matmul_1_readvariableop_resource:
цљP
Asequential_7_lstm_14_lstm_cell_18_biasadd_readvariableop_resource:	љT
@sequential_7_lstm_15_lstm_cell_19_matmul_readvariableop_resource:
цљU
Bsequential_7_lstm_15_lstm_cell_19_matmul_1_readvariableop_resource:	dљP
Asequential_7_lstm_15_lstm_cell_19_biasadd_readvariableop_resource:	љF
4sequential_7_dense_14_matmul_readvariableop_resource:dC
5sequential_7_dense_14_biasadd_readvariableop_resource:F
4sequential_7_dense_15_matmul_readvariableop_resource:C
5sequential_7_dense_15_biasadd_readvariableop_resource:
identityѕб,sequential_7/dense_14/BiasAdd/ReadVariableOpб+sequential_7/dense_14/MatMul/ReadVariableOpб,sequential_7/dense_15/BiasAdd/ReadVariableOpб+sequential_7/dense_15/MatMul/ReadVariableOpб8sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpб7sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOpб9sequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpбsequential_7/lstm_14/whileб8sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpб7sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOpб9sequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpбsequential_7/lstm_15/whileW
sequential_7/lstm_14/ShapeShapelstm_14_input*
T0*
_output_shapes
:r
(sequential_7/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_14/strided_sliceStridedSlice#sequential_7/lstm_14/Shape:output:01sequential_7/lstm_14/strided_slice/stack:output:03sequential_7/lstm_14/strided_slice/stack_1:output:03sequential_7/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#sequential_7/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ц▓
!sequential_7/lstm_14/zeros/packedPack+sequential_7/lstm_14/strided_slice:output:0,sequential_7/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    г
sequential_7/lstm_14/zerosFill*sequential_7/lstm_14/zeros/packed:output:0)sequential_7/lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         цh
%sequential_7/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цХ
#sequential_7/lstm_14/zeros_1/packedPack+sequential_7/lstm_14/strided_slice:output:0.sequential_7/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▓
sequential_7/lstm_14/zeros_1Fill,sequential_7/lstm_14/zeros_1/packed:output:0+sequential_7/lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         цx
#sequential_7/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ъ
sequential_7/lstm_14/transpose	Transposelstm_14_input,sequential_7/lstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         n
sequential_7/lstm_14/Shape_1Shape"sequential_7/lstm_14/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_14/strided_slice_1StridedSlice%sequential_7/lstm_14/Shape_1:output:03sequential_7/lstm_14/strided_slice_1/stack:output:05sequential_7/lstm_14/strided_slice_1/stack_1:output:05sequential_7/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_14/TensorArrayV2TensorListReserve9sequential_7/lstm_14/TensorArrayV2/element_shape:output:0-sequential_7/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_14/transpose:y:0Ssequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_14/strided_slice_2StridedSlice"sequential_7/lstm_14/transpose:y:03sequential_7/lstm_14/strided_slice_2/stack:output:05sequential_7/lstm_14/strided_slice_2/stack_1:output:05sequential_7/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╣
7sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_14_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0Н
(sequential_7/lstm_14/lstm_cell_18/MatMulMatMul-sequential_7/lstm_14/strided_slice_2:output:0?sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЙ
9sequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_14_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0¤
*sequential_7/lstm_14/lstm_cell_18/MatMul_1MatMul#sequential_7/lstm_14/zeros:output:0Asequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ╦
%sequential_7/lstm_14/lstm_cell_18/addAddV22sequential_7/lstm_14/lstm_cell_18/MatMul:product:04sequential_7/lstm_14/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љи
8sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0н
)sequential_7/lstm_14/lstm_cell_18/BiasAddBiasAdd)sequential_7/lstm_14/lstm_cell_18/add:z:0@sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љs
1sequential_7/lstm_14/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :а
'sequential_7/lstm_14/lstm_cell_18/splitSplit:sequential_7/lstm_14/lstm_cell_18/split/split_dim:output:02sequential_7/lstm_14/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitЎ
)sequential_7/lstm_14/lstm_cell_18/SigmoidSigmoid0sequential_7/lstm_14/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цЏ
+sequential_7/lstm_14/lstm_cell_18/Sigmoid_1Sigmoid0sequential_7/lstm_14/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         ци
%sequential_7/lstm_14/lstm_cell_18/mulMul/sequential_7/lstm_14/lstm_cell_18/Sigmoid_1:y:0%sequential_7/lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         цЊ
&sequential_7/lstm_14/lstm_cell_18/ReluRelu0sequential_7/lstm_14/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цк
'sequential_7/lstm_14/lstm_cell_18/mul_1Mul-sequential_7/lstm_14/lstm_cell_18/Sigmoid:y:04sequential_7/lstm_14/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц╗
'sequential_7/lstm_14/lstm_cell_18/add_1AddV2)sequential_7/lstm_14/lstm_cell_18/mul:z:0+sequential_7/lstm_14/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цЏ
+sequential_7/lstm_14/lstm_cell_18/Sigmoid_2Sigmoid0sequential_7/lstm_14/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цљ
(sequential_7/lstm_14/lstm_cell_18/Relu_1Relu+sequential_7/lstm_14/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         ц╩
'sequential_7/lstm_14/lstm_cell_18/mul_2Mul/sequential_7/lstm_14/lstm_cell_18/Sigmoid_2:y:06sequential_7/lstm_14/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цЃ
2sequential_7/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   э
$sequential_7/lstm_14/TensorArrayV2_1TensorListReserve;sequential_7/lstm_14/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : «
sequential_7/lstm_14/whileWhile0sequential_7/lstm_14/while/loop_counter:output:06sequential_7/lstm_14/while/maximum_iterations:output:0"sequential_7/lstm_14/time:output:0-sequential_7/lstm_14/TensorArrayV2_1:handle:0#sequential_7/lstm_14/zeros:output:0%sequential_7/lstm_14/zeros_1:output:0-sequential_7/lstm_14/strided_slice_1:output:0Lsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_14_lstm_cell_18_matmul_readvariableop_resourceBsequential_7_lstm_14_lstm_cell_18_matmul_1_readvariableop_resourceAsequential_7_lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_7_lstm_14_while_body_1465932*3
cond+R)
'sequential_7_lstm_14_while_cond_1465931*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations ќ
Esequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ѓ
7sequential_7/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_14/while:output:3Nsequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0}
*sequential_7/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
$sequential_7/lstm_14/strided_slice_3StridedSlice@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_14/strided_slice_3/stack:output:05sequential_7/lstm_14/strided_slice_3/stack_1:output:05sequential_7/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskz
%sequential_7/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
 sequential_7/lstm_14/transpose_1	Transpose@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         цp
sequential_7/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
sequential_7/lstm_15/ShapeShape$sequential_7/lstm_14/transpose_1:y:0*
T0*
_output_shapes
:r
(sequential_7/lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_15/strided_sliceStridedSlice#sequential_7/lstm_15/Shape:output:01sequential_7/lstm_15/strided_slice/stack:output:03sequential_7/lstm_15/strided_slice/stack_1:output:03sequential_7/lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d▓
!sequential_7/lstm_15/zeros/packedPack+sequential_7/lstm_15/strided_slice:output:0,sequential_7/lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_15/zerosFill*sequential_7/lstm_15/zeros/packed:output:0)sequential_7/lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:         dg
%sequential_7/lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dХ
#sequential_7/lstm_15/zeros_1/packedPack+sequential_7/lstm_15/strided_slice:output:0.sequential_7/lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_15/zeros_1Fill,sequential_7/lstm_15/zeros_1/packed:output:0+sequential_7/lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:         dx
#sequential_7/lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
sequential_7/lstm_15/transpose	Transpose$sequential_7/lstm_14/transpose_1:y:0,sequential_7/lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         цn
sequential_7/lstm_15/Shape_1Shape"sequential_7/lstm_15/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_15/strided_slice_1StridedSlice%sequential_7/lstm_15/Shape_1:output:03sequential_7/lstm_15/strided_slice_1/stack:output:05sequential_7/lstm_15/strided_slice_1/stack_1:output:05sequential_7/lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_15/TensorArrayV2TensorListReserve9sequential_7/lstm_15/TensorArrayV2/element_shape:output:0-sequential_7/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Ъ
<sequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_15/transpose:y:0Ssequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
$sequential_7/lstm_15/strided_slice_2StridedSlice"sequential_7/lstm_15/transpose:y:03sequential_7/lstm_15/strided_slice_2/stack:output:05sequential_7/lstm_15/strided_slice_2/stack_1:output:05sequential_7/lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_mask║
7sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_15_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0Н
(sequential_7/lstm_15/lstm_cell_19/MatMulMatMul-sequential_7/lstm_15/strided_slice_2:output:0?sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љй
9sequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_15_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0¤
*sequential_7/lstm_15/lstm_cell_19/MatMul_1MatMul#sequential_7/lstm_15/zeros:output:0Asequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ╦
%sequential_7/lstm_15/lstm_cell_19/addAddV22sequential_7/lstm_15/lstm_cell_19/MatMul:product:04sequential_7/lstm_15/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љи
8sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0н
)sequential_7/lstm_15/lstm_cell_19/BiasAddBiasAdd)sequential_7/lstm_15/lstm_cell_19/add:z:0@sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љs
1sequential_7/lstm_15/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
'sequential_7/lstm_15/lstm_cell_19/splitSplit:sequential_7/lstm_15/lstm_cell_19/split/split_dim:output:02sequential_7/lstm_15/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitў
)sequential_7/lstm_15/lstm_cell_19/SigmoidSigmoid0sequential_7/lstm_15/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dџ
+sequential_7/lstm_15/lstm_cell_19/Sigmoid_1Sigmoid0sequential_7/lstm_15/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dХ
%sequential_7/lstm_15/lstm_cell_19/mulMul/sequential_7/lstm_15/lstm_cell_19/Sigmoid_1:y:0%sequential_7/lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:         dњ
&sequential_7/lstm_15/lstm_cell_19/ReluRelu0sequential_7/lstm_15/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         d┼
'sequential_7/lstm_15/lstm_cell_19/mul_1Mul-sequential_7/lstm_15/lstm_cell_19/Sigmoid:y:04sequential_7/lstm_15/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d║
'sequential_7/lstm_15/lstm_cell_19/add_1AddV2)sequential_7/lstm_15/lstm_cell_19/mul:z:0+sequential_7/lstm_15/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dџ
+sequential_7/lstm_15/lstm_cell_19/Sigmoid_2Sigmoid0sequential_7/lstm_15/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dЈ
(sequential_7/lstm_15/lstm_cell_19/Relu_1Relu+sequential_7/lstm_15/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         d╔
'sequential_7/lstm_15/lstm_cell_19/mul_2Mul/sequential_7/lstm_15/lstm_cell_19/Sigmoid_2:y:06sequential_7/lstm_15/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dЃ
2sequential_7/lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   s
1sequential_7/lstm_15/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ё
$sequential_7/lstm_15/TensorArrayV2_1TensorListReserve;sequential_7/lstm_15/TensorArrayV2_1/element_shape:output:0:sequential_7/lstm_15/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ф
sequential_7/lstm_15/whileWhile0sequential_7/lstm_15/while/loop_counter:output:06sequential_7/lstm_15/while/maximum_iterations:output:0"sequential_7/lstm_15/time:output:0-sequential_7/lstm_15/TensorArrayV2_1:handle:0#sequential_7/lstm_15/zeros:output:0%sequential_7/lstm_15/zeros_1:output:0-sequential_7/lstm_15/strided_slice_1:output:0Lsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_15_lstm_cell_19_matmul_readvariableop_resourceBsequential_7_lstm_15_lstm_cell_19_matmul_1_readvariableop_resourceAsequential_7_lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_7_lstm_15_while_body_1466072*3
cond+R)
'sequential_7_lstm_15_while_cond_1466071*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations ќ
Esequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   Ћ
7sequential_7/lstm_15/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_15/while:output:3Nsequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elements}
*sequential_7/lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_15/strided_slice_3StridedSlice@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_15/strided_slice_3/stack:output:05sequential_7/lstm_15/strided_slice_3/stack_1:output:05sequential_7/lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maskz
%sequential_7/lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_15/transpose_1	Transpose@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:         dp
sequential_7/lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    а
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0╝
sequential_7/dense_14/MatMulMatMul-sequential_7/lstm_15/strided_slice_3:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0и
sequential_7/dense_15/MatMulMatMul(sequential_7/dense_14/Relu:activations:03sequential_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_15/BiasAddBiasAdd&sequential_7/dense_15/MatMul:product:04sequential_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_7/dense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ю
NoOpNoOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp-^sequential_7/dense_15/BiasAdd/ReadVariableOp,^sequential_7/dense_15/MatMul/ReadVariableOp9^sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp8^sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOp:^sequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp^sequential_7/lstm_14/while9^sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp8^sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOp:^sequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp^sequential_7/lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp2\
,sequential_7/dense_15/BiasAdd/ReadVariableOp,sequential_7/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_15/MatMul/ReadVariableOp+sequential_7/dense_15/MatMul/ReadVariableOp2t
8sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp8sequential_7/lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOp7sequential_7/lstm_14/lstm_cell_18/MatMul/ReadVariableOp2v
9sequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp9sequential_7/lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp28
sequential_7/lstm_14/whilesequential_7/lstm_14/while2t
8sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp8sequential_7/lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOp7sequential_7/lstm_15/lstm_cell_19/MatMul/ReadVariableOp2v
9sequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp9sequential_7/lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp28
sequential_7/lstm_15/whilesequential_7/lstm_15/while:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
ю

Ш
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Й
╚
while_cond_1468959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1468959___redundant_placeholder05
1while_while_cond_1468959___redundant_placeholder15
1while_while_cond_1468959___redundant_placeholder25
1while_while_cond_1468959___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
Є
В
'sequential_7_lstm_15_while_cond_1466071F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3H
Dsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1_
[sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_1466071___redundant_placeholder0_
[sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_1466071___redundant_placeholder1_
[sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_1466071___redundant_placeholder2_
[sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_1466071___redundant_placeholder3'
#sequential_7_lstm_15_while_identity
Х
sequential_7/lstm_15/while/LessLess&sequential_7_lstm_15_while_placeholderDsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_15/while/IdentityIdentity#sequential_7/lstm_15/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_15_while_identity,sequential_7/lstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
ќ$
В
while_body_1466602
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_19_1466626_0:
цљ/
while_lstm_cell_19_1466628_0:	dљ+
while_lstm_cell_19_1466630_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_19_1466626:
цљ-
while_lstm_cell_19_1466628:	dљ)
while_lstm_cell_19_1466630:	љѕб*while/lstm_cell_19/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0и
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_1466626_0while_lstm_cell_19_1466628_0while_lstm_cell_19_1466630_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466587r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         dљ
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         dy

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_19_1466626while_lstm_cell_19_1466626_0":
while_lstm_cell_19_1466628while_lstm_cell_19_1466628_0":
while_lstm_cell_19_1466630while_lstm_cell_19_1466630_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
Э
Э
.__inference_lstm_cell_18_layer_call_fn_1469741

inputs
states_0
states_1
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identity

identity_1

identity_2ѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466383p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         цr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         цr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_1
┌
є
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466735

inputs

states
states_12
matmul_readvariableop_resource:
цљ3
 matmul_1_readvariableop_resource:	dљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         dЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_namestates:OK
'
_output_shapes
:         d
 
_user_specified_namestates
ћ
╣
)__inference_lstm_15_layer_call_fn_1469055
inputs_0
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1466672o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ц
"
_user_specified_name
inputs_0
Ъ

У
lstm_14_while_cond_1467892,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1E
Alstm_14_while_lstm_14_while_cond_1467892___redundant_placeholder0E
Alstm_14_while_lstm_14_while_cond_1467892___redundant_placeholder1E
Alstm_14_while_lstm_14_while_cond_1467892___redundant_placeholder2E
Alstm_14_while_lstm_14_while_cond_1467892___redundant_placeholder3
lstm_14_while_identity
ѓ
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: [
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
Г9
М
while_body_1469148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
В
є
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466383

inputs

states
states_11
matmul_readvariableop_resource:	љ4
 matmul_1_readvariableop_resource:
цљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         цW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         цV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         цO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ц`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         цU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         цW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         цL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         цd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         цY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         цЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ц
 
_user_specified_namestates:PL
(
_output_shapes
:         ц
 
_user_specified_namestates
╚	
Ш
E__inference_dense_15_layer_call_and_return_conditional_losses_1469707

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
║
╚
while_cond_1469292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1469292___redundant_placeholder05
1while_while_cond_1469292___redundant_placeholder15
1while_while_cond_1469292___redundant_placeholder25
1while_while_cond_1469292___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Г9
М
while_body_1469583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
─
Ќ
*__inference_dense_15_layer_call_fn_1469697

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
║
╚
while_cond_1466601
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1466601___redundant_placeholder05
1while_while_cond_1466601___redundant_placeholder15
1while_while_cond_1466601___redundant_placeholder25
1while_while_cond_1466601___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Б8
М
while_body_1468960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
▓K
Ю
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467175

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1467090*
condR
while_cond_1467089*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
▓K
Ю
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467418

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1467333*
condR
while_cond_1467332*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
║
╚
while_cond_1469437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1469437___redundant_placeholder05
1while_while_cond_1469437___redundant_placeholder15
1while_while_cond_1469437___redundant_placeholder25
1while_while_cond_1469437___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
┴J
Ю
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467023

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1466939*
condR
while_cond_1466938*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
­
╠
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467647

inputs"
lstm_14_1467622:	љ#
lstm_14_1467624:
цљ
lstm_14_1467626:	љ#
lstm_15_1467629:
цљ"
lstm_15_1467631:	dљ
lstm_15_1467633:	љ"
dense_14_1467636:d
dense_14_1467638:"
dense_15_1467641:
dense_15_1467643:
identityѕб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallбlstm_14/StatefulPartitionedCallбlstm_15/StatefulPartitionedCallЄ
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_1467622lstm_14_1467624lstm_14_1467626*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467583ц
lstm_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0lstm_15_1467629lstm_15_1467631lstm_15_1467633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467418Ћ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_14_1467636dense_14_1467638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194ќ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_1467641dense_15_1467643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
«
╣
)__inference_lstm_14_layer_call_fn_1468450
inputs_0
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1466511}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
ќ$
В
while_body_1466795
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_19_1466819_0:
цљ/
while_lstm_cell_19_1466821_0:	dљ+
while_lstm_cell_19_1466823_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_19_1466819:
цљ-
while_lstm_cell_19_1466821:	dљ)
while_lstm_cell_19_1466823:	љѕб*while/lstm_cell_19/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0и
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_1466819_0while_lstm_cell_19_1466821_0while_lstm_cell_19_1466823_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466735r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         dљ
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         dy

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_19_1466819while_lstm_cell_19_1466819_0":
while_lstm_cell_19_1466821while_lstm_cell_19_1466821_0":
while_lstm_cell_19_1466823while_lstm_cell_19_1466823_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
З
ѕ
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469773

inputs
states_0
states_11
matmul_readvariableop_resource:	љ4
 matmul_1_readvariableop_resource:
цљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         цW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         цV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         цO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ц`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         цU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         цW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         цL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         цd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         цY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         цЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_1
║
╚
while_cond_1469147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1469147___redundant_placeholder05
1while_while_cond_1469147___redundant_placeholder15
1while_while_cond_1469147___redundant_placeholder25
1while_while_cond_1469147___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Й
╚
while_cond_1466938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1466938___redundant_placeholder05
1while_while_cond_1466938___redundant_placeholder15
1while_while_cond_1466938___redundant_placeholder25
1while_while_cond_1466938___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
­
Э
.__inference_lstm_cell_19_layer_call_fn_1469839

inputs
states_0
states_1
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466735o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states_0:QM
'
_output_shapes
:         d
"
_user_specified_name
states_1
НK
Ъ
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469378
inputs_0?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1469293*
condR
while_cond_1469292*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ц
"
_user_specified_name
inputs_0
ћ
╣
)__inference_lstm_15_layer_call_fn_1469066
inputs_0
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1466865o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ц
"
_user_specified_name
inputs_0
ѓ#
В
while_body_1466251
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_18_1466275_0:	љ0
while_lstm_cell_18_1466277_0:
цљ+
while_lstm_cell_18_1466279_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_18_1466275:	љ.
while_lstm_cell_18_1466277:
цљ)
while_lstm_cell_18_1466279:	љѕб*while/lstm_cell_18/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0║
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_1466275_0while_lstm_cell_18_1466277_0while_lstm_cell_18_1466279_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466237▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Љ
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         цЉ
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         цy

while/NoOpNoOp+^while/lstm_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_18_1466275while_lstm_cell_18_1466275_0":
while_lstm_cell_18_1466277while_lstm_cell_18_1466277_0":
while_lstm_cell_18_1466279while_lstm_cell_18_1466279_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
┴J
Ю
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468901

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1468817*
condR
while_cond_1468816*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▓K
Ю
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469668

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
цљ@
-lstm_cell_19_matmul_1_readvariableop_resource:	dљ;
,lstm_cell_19_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_19/BiasAdd/ReadVariableOpб"lstm_cell_19/MatMul/ReadVariableOpб$lstm_cell_19/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskљ
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0ќ
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0љ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dw
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         dh
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dє
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dp
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:         de
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dі
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1469583*
condR
while_cond_1469582*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d└
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
Й
╚
while_cond_1468530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1468530___redundant_placeholder05
1while_while_cond_1468530___redundant_placeholder15
1while_while_cond_1468530___redundant_placeholder25
1while_while_cond_1468530___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
Ч
и
)__inference_lstm_15_layer_call_fn_1469077

inputs
unknown:
цљ
	unknown_0:	dљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ц
 
_user_specified_nameinputs
ѓ#
В
while_body_1466442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_18_1466466_0:	љ0
while_lstm_cell_18_1466468_0:
цљ+
while_lstm_cell_18_1466470_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_18_1466466:	љ.
while_lstm_cell_18_1466468:
цљ)
while_lstm_cell_18_1466470:	љѕб*while/lstm_cell_18/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0║
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_1466466_0while_lstm_cell_18_1466468_0while_lstm_cell_18_1466470_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466383▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Љ
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         цЉ
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         цy

while/NoOpNoOp+^while/lstm_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_18_1466466while_lstm_cell_18_1466466_0":
while_lstm_cell_18_1466468while_lstm_cell_18_1466468_0":
while_lstm_cell_18_1466470while_lstm_cell_18_1466470_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
рх
И	
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468131

inputsF
3lstm_14_lstm_cell_18_matmul_readvariableop_resource:	љI
5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource:
цљC
4lstm_14_lstm_cell_18_biasadd_readvariableop_resource:	љG
3lstm_15_lstm_cell_19_matmul_readvariableop_resource:
цљH
5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource:	dљC
4lstm_15_lstm_cell_19_biasadd_readvariableop_resource:	љ9
'dense_14_matmul_readvariableop_resource:d6
(dense_14_biasadd_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбdense_15/BiasAdd/ReadVariableOpбdense_15/MatMul/ReadVariableOpб+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpб*lstm_14/lstm_cell_18/MatMul/ReadVariableOpб,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpбlstm_14/whileб+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpб*lstm_15/lstm_cell_19/MatMul/ReadVariableOpб,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpбlstm_15/whileC
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цІ
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         ц[
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цЈ
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         цk
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:g
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЪ
*lstm_14/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0«
lstm_14/lstm_cell_18/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0е
lstm_14/lstm_cell_18/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
lstm_14/lstm_cell_18/addAddV2%lstm_14/lstm_cell_18/MatMul:product:0'lstm_14/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЮ
+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Г
lstm_14/lstm_cell_18/BiasAddBiasAddlstm_14/lstm_cell_18/add:z:03lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љf
$lstm_14/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :щ
lstm_14/lstm_cell_18/splitSplit-lstm_14/lstm_cell_18/split/split_dim:output:0%lstm_14/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split
lstm_14/lstm_cell_18/SigmoidSigmoid#lstm_14/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цЂ
lstm_14/lstm_cell_18/Sigmoid_1Sigmoid#lstm_14/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цљ
lstm_14/lstm_cell_18/mulMul"lstm_14/lstm_cell_18/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         цy
lstm_14/lstm_cell_18/ReluRelu#lstm_14/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЪ
lstm_14/lstm_cell_18/mul_1Mul lstm_14/lstm_cell_18/Sigmoid:y:0'lstm_14/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цћ
lstm_14/lstm_cell_18/add_1AddV2lstm_14/lstm_cell_18/mul:z:0lstm_14/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цЂ
lstm_14/lstm_cell_18/Sigmoid_2Sigmoid#lstm_14/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цv
lstm_14/lstm_cell_18/Relu_1Relulstm_14/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цБ
lstm_14/lstm_cell_18/mul_2Mul"lstm_14/lstm_cell_18/Sigmoid_2:y:0)lstm_14/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цv
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   л
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_18_matmul_readvariableop_resource5lstm_14_lstm_cell_18_matmul_1_readvariableop_resource4lstm_14_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_14_while_body_1467893*&
condR
lstm_14_while_cond_1467892*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ѕ
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   █
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0p
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskm
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         цc
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_15/ShapeShapelstm_14/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dІ
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:         dZ
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЈ
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:         dk
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
lstm_15/transpose	Transposelstm_14/transpose_1:y:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         цT
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:g
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Э
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskа
*lstm_15/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0«
lstm_15/lstm_cell_19/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љБ
,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0е
lstm_15/lstm_cell_19/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љц
lstm_15/lstm_cell_19/addAddV2%lstm_15/lstm_cell_19/MatMul:product:0'lstm_15/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЮ
+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Г
lstm_15/lstm_cell_19/BiasAddBiasAddlstm_15/lstm_cell_19/add:z:03lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љf
$lstm_15/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_15/lstm_cell_19/splitSplit-lstm_15/lstm_cell_19/split/split_dim:output:0%lstm_15/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_split~
lstm_15/lstm_cell_19/SigmoidSigmoid#lstm_15/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dђ
lstm_15/lstm_cell_19/Sigmoid_1Sigmoid#lstm_15/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dЈ
lstm_15/lstm_cell_19/mulMul"lstm_15/lstm_cell_19/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:         dx
lstm_15/lstm_cell_19/ReluRelu#lstm_15/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dъ
lstm_15/lstm_cell_19/mul_1Mul lstm_15/lstm_cell_19/Sigmoid:y:0'lstm_15/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЊ
lstm_15/lstm_cell_19/add_1AddV2lstm_15/lstm_cell_19/mul:z:0lstm_15/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dђ
lstm_15/lstm_cell_19/Sigmoid_2Sigmoid#lstm_15/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         du
lstm_15/lstm_cell_19/Relu_1Relulstm_15/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dб
lstm_15/lstm_cell_19/mul_2Mul"lstm_15/lstm_cell_19/Sigmoid_2:y:0)lstm_15/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dv
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   f
$lstm_15/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0-lstm_15/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : З
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_19_matmul_readvariableop_resource5lstm_15_lstm_cell_19_matmul_1_readvariableop_resource4lstm_15_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_15_while_body_1468033*&
condR
lstm_15_while_cond_1468032*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ѕ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   Ь
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsp
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maskm
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:         dc
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    є
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ћ
dense_14/MatMulMatMul lstm_15/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ђ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp,^lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_18/MatMul/ReadVariableOp-^lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_19/MatMul/ReadVariableOp-^lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2Z
+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_18/MatMul/ReadVariableOp*lstm_14/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while2Z
+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_15/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_15/lstm_cell_19/MatMul/ReadVariableOp*lstm_15/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_15/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┴J
Ю
D__inference_lstm_14_layer_call_and_return_conditional_losses_1469044

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1468960*
condR
while_cond_1468959*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ЬB
М

lstm_15_while_body_1468330,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0:
цљP
=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљK
<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_19_matmul_readvariableop_resource:
цљN
;lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource:	dљI
:lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpб0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpб2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpљ
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ¤
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0«
0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0м
!lstm_15/while/lstm_cell_19/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ▒
2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0╣
#lstm_15/while/lstm_cell_19/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љХ
lstm_15/while/lstm_cell_19/addAddV2+lstm_15/while/lstm_cell_19/MatMul:product:0-lstm_15/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љФ
1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0┐
"lstm_15/while/lstm_cell_19/BiasAddBiasAdd"lstm_15/while/lstm_cell_19/add:z:09lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љl
*lstm_15/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_15/while/lstm_cell_19/splitSplit3lstm_15/while/lstm_cell_19/split/split_dim:output:0+lstm_15/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitі
"lstm_15/while/lstm_cell_19/SigmoidSigmoid)lstm_15/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dї
$lstm_15/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dъ
lstm_15/while/lstm_cell_19/mulMul(lstm_15/while/lstm_cell_19/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:         dё
lstm_15/while/lstm_cell_19/ReluRelu)lstm_15/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         d░
 lstm_15/while/lstm_cell_19/mul_1Mul&lstm_15/while/lstm_cell_19/Sigmoid:y:0-lstm_15/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЦ
 lstm_15/while/lstm_cell_19/add_1AddV2"lstm_15/while/lstm_cell_19/mul:z:0$lstm_15/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dї
$lstm_15/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dЂ
!lstm_15/while/lstm_cell_19/Relu_1Relu$lstm_15/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         d┤
 lstm_15/while/lstm_cell_19/mul_2Mul(lstm_15/while/lstm_cell_19/Sigmoid_2:y:0/lstm_15/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dz
8lstm_15/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1Alstm_15/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_15/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: і
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: q
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: ъ
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: Љ
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_19/mul_2:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:         dЉ
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_19/add_1:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:         d­
lstm_15/while/NoOpNoOp2^lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"z
:lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_15_while_lstm_cell_19_matmul_readvariableop_resource;lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0"╚
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2f
1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
┼

є
.__inference_sequential_7_layer_call_fn_1467240
lstm_14_input
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
	unknown_2:
цљ
	unknown_3:	dљ
	unknown_4:	љ
	unknown_5:d
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
цR
З
'sequential_7_lstm_15_while_body_1466072F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3E
Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0Ђ
}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_7_lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0:
цљ]
Jsequential_7_lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљX
Isequential_7_lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ'
#sequential_7_lstm_15_while_identity)
%sequential_7_lstm_15_while_identity_1)
%sequential_7_lstm_15_while_identity_2)
%sequential_7_lstm_15_while_identity_3)
%sequential_7_lstm_15_while_identity_4)
%sequential_7_lstm_15_while_identity_5C
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_7_lstm_15_while_lstm_cell_19_matmul_readvariableop_resource:
цљ[
Hsequential_7_lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource:	dљV
Gsequential_7_lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб>sequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpб=sequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpб?sequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   љ
>sequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_15_while_placeholderUsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0╚
=sequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0щ
.sequential_7/lstm_15/while/lstm_cell_19/MatMulMatMulEsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ╦
?sequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0Я
0sequential_7/lstm_15/while/lstm_cell_19/MatMul_1MatMul(sequential_7_lstm_15_while_placeholder_2Gsequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љП
+sequential_7/lstm_15/while/lstm_cell_19/addAddV28sequential_7/lstm_15/while/lstm_cell_19/MatMul:product:0:sequential_7/lstm_15/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љ┼
>sequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Т
/sequential_7/lstm_15/while/lstm_cell_19/BiasAddBiasAdd/sequential_7/lstm_15/while/lstm_cell_19/add:z:0Fsequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
7sequential_7/lstm_15/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
-sequential_7/lstm_15/while/lstm_cell_19/splitSplit@sequential_7/lstm_15/while/lstm_cell_19/split/split_dim:output:08sequential_7/lstm_15/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitц
/sequential_7/lstm_15/while/lstm_cell_19/SigmoidSigmoid6sequential_7/lstm_15/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         dд
1sequential_7/lstm_15/while/lstm_cell_19/Sigmoid_1Sigmoid6sequential_7/lstm_15/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         d┼
+sequential_7/lstm_15/while/lstm_cell_19/mulMul5sequential_7/lstm_15/while/lstm_cell_19/Sigmoid_1:y:0(sequential_7_lstm_15_while_placeholder_3*
T0*'
_output_shapes
:         dъ
,sequential_7/lstm_15/while/lstm_cell_19/ReluRelu6sequential_7/lstm_15/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dО
-sequential_7/lstm_15/while/lstm_cell_19/mul_1Mul3sequential_7/lstm_15/while/lstm_cell_19/Sigmoid:y:0:sequential_7/lstm_15/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         d╠
-sequential_7/lstm_15/while/lstm_cell_19/add_1AddV2/sequential_7/lstm_15/while/lstm_cell_19/mul:z:01sequential_7/lstm_15/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         dд
1sequential_7/lstm_15/while/lstm_cell_19/Sigmoid_2Sigmoid6sequential_7/lstm_15/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dЏ
.sequential_7/lstm_15/while/lstm_cell_19/Relu_1Relu1sequential_7/lstm_15/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         d█
-sequential_7/lstm_15/while/lstm_cell_19/mul_2Mul5sequential_7/lstm_15/while/lstm_cell_19/Sigmoid_2:y:0<sequential_7/lstm_15/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dЄ
Esequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ┴
?sequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_15_while_placeholder_1Nsequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem/index:output:01sequential_7/lstm_15/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_15/while/addAddV2&sequential_7_lstm_15_while_placeholder)sequential_7/lstm_15/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_15/while/add_1AddV2Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counter+sequential_7/lstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_15/while/IdentityIdentity$sequential_7/lstm_15/while/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_15/while/Identity_1IdentityHsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_15/while/Identity_2Identity"sequential_7/lstm_15/while/add:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_15/while/Identity_3IdentityOsequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: И
%sequential_7/lstm_15/while/Identity_4Identity1sequential_7/lstm_15/while/lstm_cell_19/mul_2:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*'
_output_shapes
:         dИ
%sequential_7/lstm_15/while/Identity_5Identity1sequential_7/lstm_15/while/lstm_cell_19/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*'
_output_shapes
:         dц
sequential_7/lstm_15/while/NoOpNoOp?^sequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp>^sequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp@^sequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_15_while_identity,sequential_7/lstm_15/while/Identity:output:0"W
%sequential_7_lstm_15_while_identity_1.sequential_7/lstm_15/while/Identity_1:output:0"W
%sequential_7_lstm_15_while_identity_2.sequential_7/lstm_15/while/Identity_2:output:0"W
%sequential_7_lstm_15_while_identity_3.sequential_7/lstm_15/while/Identity_3:output:0"W
%sequential_7_lstm_15_while_identity_4.sequential_7/lstm_15/while/Identity_4:output:0"W
%sequential_7_lstm_15_while_identity_5.sequential_7/lstm_15/while/Identity_5:output:0"ћ
Gsequential_7_lstm_15_while_lstm_cell_19_biasadd_readvariableop_resourceIsequential_7_lstm_15_while_lstm_cell_19_biasadd_readvariableop_resource_0"ќ
Hsequential_7_lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resourceJsequential_7_lstm_15_while_lstm_cell_19_matmul_1_readvariableop_resource_0"њ
Fsequential_7_lstm_15_while_lstm_cell_19_matmul_readvariableop_resourceHsequential_7_lstm_15_while_lstm_cell_19_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0"Ч
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2ђ
>sequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp>sequential_7/lstm_15/while/lstm_cell_19/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp=sequential_7/lstm_15/while/lstm_cell_19/MatMul/ReadVariableOp2ѓ
?sequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp?sequential_7/lstm_15/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
▄A
М

lstm_14_while_body_1467893,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0:	љQ
=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљK
<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_18_matmul_readvariableop_resource:	љO
;lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource:
цљI
:lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpб0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpб2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpљ
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Г
0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0м
!lstm_14/while/lstm_cell_18/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ▓
2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0╣
#lstm_14/while/lstm_cell_18/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љХ
lstm_14/while/lstm_cell_18/addAddV2+lstm_14/while/lstm_cell_18/MatMul:product:0-lstm_14/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љФ
1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0┐
"lstm_14/while/lstm_cell_18/BiasAddBiasAdd"lstm_14/while/lstm_cell_18/add:z:09lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љl
*lstm_14/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :І
 lstm_14/while/lstm_cell_18/splitSplit3lstm_14/while/lstm_cell_18/split/split_dim:output:0+lstm_14/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitІ
"lstm_14/while/lstm_cell_18/SigmoidSigmoid)lstm_14/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цЇ
$lstm_14/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЪ
lstm_14/while/lstm_cell_18/mulMul(lstm_14/while/lstm_cell_18/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         цЁ
lstm_14/while/lstm_cell_18/ReluRelu)lstm_14/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         ц▒
 lstm_14/while/lstm_cell_18/mul_1Mul&lstm_14/while/lstm_cell_18/Sigmoid:y:0-lstm_14/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цд
 lstm_14/while/lstm_cell_18/add_1AddV2"lstm_14/while/lstm_cell_18/mul:z:0$lstm_14/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цЇ
$lstm_14/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цѓ
!lstm_14/while/lstm_cell_18/Relu_1Relu$lstm_14/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цх
 lstm_14/while/lstm_cell_18/mul_2Mul(lstm_14/while/lstm_cell_18/Sigmoid_2:y:0/lstm_14/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цт
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: і
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: q
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: ъ
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: њ
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_18/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         цњ
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_18/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ц­
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_18_matmul_readvariableop_resource;lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0"╚
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2f
1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
░

 
.__inference_sequential_7_layer_call_fn_1467834

inputs
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
	unknown_2:
цљ
	unknown_3:	dљ
	unknown_4:	љ
	unknown_5:d
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
­
╠
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467217

inputs"
lstm_14_1467024:	љ#
lstm_14_1467026:
цљ
lstm_14_1467028:	љ#
lstm_15_1467176:
цљ"
lstm_15_1467178:	dљ
lstm_15_1467180:	љ"
dense_14_1467195:d
dense_14_1467197:"
dense_15_1467211:
dense_15_1467213:
identityѕб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallбlstm_14/StatefulPartitionedCallбlstm_15/StatefulPartitionedCallЄ
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_1467024lstm_14_1467026lstm_14_1467028*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467023ц
lstm_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0lstm_15_1467176lstm_15_1467178lstm_15_1467180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467175Ћ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_14_1467195dense_14_1467197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194ќ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_1467211dense_15_1467213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
░

 
.__inference_sequential_7_layer_call_fn_1467809

inputs
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
	unknown_2:
цљ
	unknown_3:	dљ
	unknown_4:	љ
	unknown_5:d
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Б8
М
while_body_1468674
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
ёQ
З
'sequential_7_lstm_14_while_body_1465932F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3E
Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0Ђ
}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0:	љ^
Jsequential_7_lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљX
Isequential_7_lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ'
#sequential_7_lstm_14_while_identity)
%sequential_7_lstm_14_while_identity_1)
%sequential_7_lstm_14_while_identity_2)
%sequential_7_lstm_14_while_identity_3)
%sequential_7_lstm_14_while_identity_4)
%sequential_7_lstm_14_while_identity_5C
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_14_while_lstm_cell_18_matmul_readvariableop_resource:	љ\
Hsequential_7_lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource:
цљV
Gsequential_7_lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб>sequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpб=sequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpб?sequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_14_while_placeholderUsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0К
=sequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0щ
.sequential_7/lstm_14/while/lstm_cell_18/MatMulMatMulEsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ╠
?sequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0Я
0sequential_7/lstm_14/while/lstm_cell_18/MatMul_1MatMul(sequential_7_lstm_14_while_placeholder_2Gsequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љП
+sequential_7/lstm_14/while/lstm_cell_18/addAddV28sequential_7/lstm_14/while/lstm_cell_18/MatMul:product:0:sequential_7/lstm_14/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љ┼
>sequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Т
/sequential_7/lstm_14/while/lstm_cell_18/BiasAddBiasAdd/sequential_7/lstm_14/while/lstm_cell_18/add:z:0Fsequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
7sequential_7/lstm_14/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▓
-sequential_7/lstm_14/while/lstm_cell_18/splitSplit@sequential_7/lstm_14/while/lstm_cell_18/split/split_dim:output:08sequential_7/lstm_14/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitЦ
/sequential_7/lstm_14/while/lstm_cell_18/SigmoidSigmoid6sequential_7/lstm_14/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цД
1sequential_7/lstm_14/while/lstm_cell_18/Sigmoid_1Sigmoid6sequential_7/lstm_14/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цк
+sequential_7/lstm_14/while/lstm_cell_18/mulMul5sequential_7/lstm_14/while/lstm_cell_18/Sigmoid_1:y:0(sequential_7_lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         цЪ
,sequential_7/lstm_14/while/lstm_cell_18/ReluRelu6sequential_7/lstm_14/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цп
-sequential_7/lstm_14/while/lstm_cell_18/mul_1Mul3sequential_7/lstm_14/while/lstm_cell_18/Sigmoid:y:0:sequential_7/lstm_14/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц═
-sequential_7/lstm_14/while/lstm_cell_18/add_1AddV2/sequential_7/lstm_14/while/lstm_cell_18/mul:z:01sequential_7/lstm_14/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цД
1sequential_7/lstm_14/while/lstm_cell_18/Sigmoid_2Sigmoid6sequential_7/lstm_14/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цю
.sequential_7/lstm_14/while/lstm_cell_18/Relu_1Relu1sequential_7/lstm_14/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         ц▄
-sequential_7/lstm_14/while/lstm_cell_18/mul_2Mul5sequential_7/lstm_14/while/lstm_cell_18/Sigmoid_2:y:0<sequential_7/lstm_14/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цЎ
?sequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_14_while_placeholder_1&sequential_7_lstm_14_while_placeholder1sequential_7/lstm_14/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_14/while/addAddV2&sequential_7_lstm_14_while_placeholder)sequential_7/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_14/while/add_1AddV2Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counter+sequential_7/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_14/while/IdentityIdentity$sequential_7/lstm_14/while/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_14/while/Identity_1IdentityHsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_14/while/Identity_2Identity"sequential_7/lstm_14/while/add:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_14/while/Identity_3IdentityOsequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: ╣
%sequential_7/lstm_14/while/Identity_4Identity1sequential_7/lstm_14/while/lstm_cell_18/mul_2:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:         ц╣
%sequential_7/lstm_14/while/Identity_5Identity1sequential_7/lstm_14/while/lstm_cell_18/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:         цц
sequential_7/lstm_14/while/NoOpNoOp?^sequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp>^sequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp@^sequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_14_while_identity,sequential_7/lstm_14/while/Identity:output:0"W
%sequential_7_lstm_14_while_identity_1.sequential_7/lstm_14/while/Identity_1:output:0"W
%sequential_7_lstm_14_while_identity_2.sequential_7/lstm_14/while/Identity_2:output:0"W
%sequential_7_lstm_14_while_identity_3.sequential_7/lstm_14/while/Identity_3:output:0"W
%sequential_7_lstm_14_while_identity_4.sequential_7/lstm_14/while/Identity_4:output:0"W
%sequential_7_lstm_14_while_identity_5.sequential_7/lstm_14/while/Identity_5:output:0"ћ
Gsequential_7_lstm_14_while_lstm_cell_18_biasadd_readvariableop_resourceIsequential_7_lstm_14_while_lstm_cell_18_biasadd_readvariableop_resource_0"ќ
Hsequential_7_lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resourceJsequential_7_lstm_14_while_lstm_cell_18_matmul_1_readvariableop_resource_0"њ
Fsequential_7_lstm_14_while_lstm_cell_18_matmul_readvariableop_resourceHsequential_7_lstm_14_while_lstm_cell_18_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0"Ч
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2ђ
>sequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp>sequential_7/lstm_14/while/lstm_cell_18/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp=sequential_7/lstm_14/while/lstm_cell_18/MatMul/ReadVariableOp2ѓ
?sequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp?sequential_7/lstm_14/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
Г9
М
while_body_1467090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
ЪQ
┴
 __inference__traced_save_1470037
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop:
6savev2_lstm_14_lstm_cell_18_kernel_read_readvariableopD
@savev2_lstm_14_lstm_cell_18_recurrent_kernel_read_readvariableop8
4savev2_lstm_14_lstm_cell_18_bias_read_readvariableop:
6savev2_lstm_15_lstm_cell_19_kernel_read_readvariableopD
@savev2_lstm_15_lstm_cell_19_recurrent_kernel_read_readvariableop8
4savev2_lstm_15_lstm_cell_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_18_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_18_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_18_bias_m_read_readvariableopA
=savev2_adam_lstm_15_lstm_cell_19_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_15_lstm_cell_19_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_15_lstm_cell_19_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_18_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_18_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_18_bias_v_read_readvariableopA
=savev2_adam_lstm_15_lstm_cell_19_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_15_lstm_cell_19_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_15_lstm_cell_19_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: №
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ў
valueјBІ&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╣
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B й
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop6savev2_lstm_14_lstm_cell_18_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_18_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_18_bias_read_readvariableop6savev2_lstm_15_lstm_cell_19_kernel_read_readvariableop@savev2_lstm_15_lstm_cell_19_recurrent_kernel_read_readvariableop4savev2_lstm_15_lstm_cell_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop=savev2_adam_lstm_14_lstm_cell_18_kernel_m_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_18_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_14_lstm_cell_18_bias_m_read_readvariableop=savev2_adam_lstm_15_lstm_cell_19_kernel_m_read_readvariableopGsavev2_adam_lstm_15_lstm_cell_19_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_15_lstm_cell_19_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop=savev2_adam_lstm_14_lstm_cell_18_kernel_v_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_18_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_14_lstm_cell_18_bias_v_read_readvariableop=savev2_adam_lstm_15_lstm_cell_19_kernel_v_read_readvariableopGsavev2_adam_lstm_15_lstm_cell_19_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_15_lstm_cell_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *4
dtypes*
(2&	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╗
_input_shapesЕ
д: :d::::	љ:
цљ:љ:
цљ:	dљ:љ: : : : : : : :d::::	љ:
цљ:љ:
цљ:	dљ:љ:d::::	љ:
цљ:љ:
цљ:	dљ:љ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	љ:&"
 
_output_shapes
:
цљ:!

_output_shapes	
:љ:&"
 
_output_shapes
:
цљ:%	!

_output_shapes
:	dљ:!


_output_shapes	
:љ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	љ:&"
 
_output_shapes
:
цљ:!

_output_shapes	
:љ:&"
 
_output_shapes
:
цљ:%!

_output_shapes
:	dљ:!

_output_shapes	
:љ:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::% !

_output_shapes
:	љ:&!"
 
_output_shapes
:
цљ:!"

_output_shapes	
:љ:&#"
 
_output_shapes
:
цљ:%$!

_output_shapes
:	dљ:!%

_output_shapes	
:љ:&

_output_shapes
: 
Ј9
і
D__inference_lstm_15_layer_call_and_return_conditional_losses_1466672

inputs(
lstm_cell_19_1466588:
цљ'
lstm_cell_19_1466590:	dљ#
lstm_cell_19_1466592:	љ
identityѕб$lstm_cell_19/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  цD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maskщ
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_1466588lstm_cell_19_1466590lstm_cell_19_1466592*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         d:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466587n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_1466588lstm_cell_19_1466590lstm_cell_19_1466592*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         d:         d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1466602*
condR
while_cond_1466601*K
output_shapes:
8: : : : :         d:         d: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         du
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ц: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ц
 
_user_specified_nameinputs
║
╚
while_cond_1467332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1467332___redundant_placeholder05
1while_while_cond_1467332___redundant_placeholder15
1while_while_cond_1467332___redundant_placeholder25
1while_while_cond_1467332___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Ё
М
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467723
lstm_14_input"
lstm_14_1467698:	љ#
lstm_14_1467700:
цљ
lstm_14_1467702:	љ#
lstm_15_1467705:
цљ"
lstm_15_1467707:	dљ
lstm_15_1467709:	љ"
dense_14_1467712:d
dense_14_1467714:"
dense_15_1467717:
dense_15_1467719:
identityѕб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallбlstm_14/StatefulPartitionedCallбlstm_15/StatefulPartitionedCallј
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_1467698lstm_14_1467700lstm_14_1467702*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467023ц
lstm_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0lstm_15_1467705lstm_15_1467707lstm_15_1467709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467175Ћ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_14_1467712dense_14_1467714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194ќ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_1467717dense_15_1467719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
Й
╚
while_cond_1466441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1466441___redundant_placeholder05
1while_while_cond_1466441___redundant_placeholder15
1while_while_cond_1466441___redundant_placeholder25
1while_while_cond_1466441___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
»8
і
D__inference_lstm_14_layer_call_and_return_conditional_losses_1466320

inputs'
lstm_cell_18_1466238:	љ(
lstm_cell_18_1466240:
цљ#
lstm_cell_18_1466242:	љ
identityѕб$lstm_cell_18/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЧ
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_1466238lstm_cell_18_1466240lstm_cell_18_1466242*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ц:         ц:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1466237n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : └
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_1466238lstm_cell_18_1466240lstm_cell_18_1466242*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1466251*
condR
while_cond_1466250*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  цu
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Џ

У
lstm_15_while_cond_1468329,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1E
Alstm_15_while_lstm_15_while_cond_1468329___redundant_placeholder0E
Alstm_15_while_lstm_15_while_cond_1468329___redundant_placeholder1E
Alstm_15_while_lstm_15_while_cond_1468329___redundant_placeholder2E
Alstm_15_while_lstm_15_while_cond_1468329___redundant_placeholder3
lstm_15_while_identity
ѓ
lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: [
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Ъ

У
lstm_14_while_cond_1468189,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1E
Alstm_14_while_lstm_14_while_cond_1468189___redundant_placeholder0E
Alstm_14_while_lstm_14_while_cond_1468189___redundant_placeholder1E
Alstm_14_while_lstm_14_while_cond_1468189___redundant_placeholder2E
Alstm_14_while_lstm_14_while_cond_1468189___redundant_placeholder3
lstm_14_while_identity
ѓ
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: [
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
┌
є
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1466587

inputs

states
states_12
matmul_readvariableop_resource:
цљ3
 matmul_1_readvariableop_resource:	dљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         dЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_namestates:OK
'
_output_shapes
:         d
 
_user_specified_namestates
Г9
М
while_body_1469293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
Г9
М
while_body_1467333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
цљH
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:	dљC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
цљF
3while_lstm_cell_19_matmul_1_readvariableop_resource:	dљA
2while_lstm_cell_19_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_19/BiasAdd/ReadVariableOpб(while/lstm_cell_19/MatMul/ReadVariableOpб*while/lstm_cell_19/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   Д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ц*
element_dtype0ъ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0║
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љА
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	dљ*
dtype0А
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:         dє
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         dt
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:         dў
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:         dЇ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:         d|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:         dq
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:         dю
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:         dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         dy
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         dл

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         d:         d: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
: 
Р
ѕ
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469903

inputs
states_0
states_12
matmul_readvariableop_resource:
цљ3
 matmul_1_readvariableop_resource:	dљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dљ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         d:         d:         d:         d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         dЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         ц:         d:         d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ц
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states_0:QM
'
_output_shapes
:         d
"
_user_specified_name
states_1
Й
╚
while_cond_1467498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1467498___redundant_placeholder05
1while_while_cond_1467498___redundant_placeholder15
1while_while_cond_1467498___redundant_placeholder25
1while_while_cond_1467498___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
З
ѕ
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469805

inputs
states_0
states_11
matmul_readvariableop_resource:	љ4
 matmul_1_readvariableop_resource:
цљ.
biasadd_readvariableop_resource:	љ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         цW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         цV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         цO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ц`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         цU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         цW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         цL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         цd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         цY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ц[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         цЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ц:         ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ц
"
_user_specified_name
states_1
Ё
М
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467751
lstm_14_input"
lstm_14_1467726:	љ#
lstm_14_1467728:
цљ
lstm_14_1467730:	љ#
lstm_15_1467733:
цљ"
lstm_15_1467735:	dљ
lstm_15_1467737:	љ"
dense_14_1467740:d
dense_14_1467742:"
dense_15_1467745:
dense_15_1467747:
identityѕб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallбlstm_14/StatefulPartitionedCallбlstm_15/StatefulPartitionedCallј
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_1467726lstm_14_1467728lstm_14_1467730*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467583ц
lstm_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0lstm_15_1467733lstm_15_1467735lstm_15_1467737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_15_layer_call_and_return_conditional_losses_1467418Ћ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0dense_14_1467740dense_14_1467742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1467194ќ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_1467745dense_15_1467747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1467210x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_14_input
║
╚
while_cond_1467089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1467089___redundant_placeholder05
1while_while_cond_1467089___redundant_placeholder15
1while_while_cond_1467089___redundant_placeholder25
1while_while_cond_1467089___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         d:         d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         d:-)
'
_output_shapes
:         d:

_output_shapes
: :

_output_shapes
:
Й
╚
while_cond_1468673
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1468673___redundant_placeholder05
1while_while_cond_1468673___redundant_placeholder15
1while_while_cond_1468673___redundant_placeholder25
1while_while_cond_1468673___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ц:         ц: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
:
┴J
Ю
D__inference_lstm_14_layer_call_and_return_conditional_losses_1467583

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	љA
-lstm_cell_18_matmul_1_readvariableop_resource:
цљ;
,lstm_cell_18_biasadd_readvariableop_resource:	љ
identityѕб#lstm_cell_18/BiasAdd/ReadVariableOpб"lstm_cell_18/MatMul/ReadVariableOpб$lstm_cell_18/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         цS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :цw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         цc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0ќ
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љћ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
цљ*
dtype0љ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љї
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЇ
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ћ
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ^
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_splito
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цx
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         цi
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЄ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         ц|
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         цq
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цf
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цІ
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         цn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ц:         ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1467499*
condR
while_cond_1467498*M
output_shapes<
:: : : : :         ц:         ц: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ц   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ц*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ц*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ц[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ц└
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Б8
М
while_body_1468817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	љI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
цљC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	љ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	љG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
цљA
2while_lstm_cell_18_biasadd_readvariableop_resource:	љѕб)while/lstm_cell_18/BiasAdd/ReadVariableOpб(while/lstm_cell_18/MatMul/ReadVariableOpб*while/lstm_cell_18/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	љ*
dtype0║
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љб
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
цљ*
dtype0А
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љъ
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         љЏ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:љ*
dtype0Д
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ц:         ц:         ц:         ц*
	num_split{
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:         цЄ
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         цu
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:         цЎ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:         цј
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:         ц}
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:         цr
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:         цЮ
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:         ц┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         цz
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         цл

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ц:         ц: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ц:.*
(
_output_shapes
:         ц:

_output_shapes
: :

_output_shapes
: 
«
╣
)__inference_lstm_14_layer_call_fn_1468439
inputs_0
unknown:	љ
	unknown_0:
цљ
	unknown_1:	љ
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_14_layer_call_and_return_conditional_losses_1466320}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ц`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultД
K
lstm_14_input:
serving_default_lstm_14_input:0         <
dense_150
StatefulPartitionedCall:0         tensorflow/serving/predict:гѓ
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╗
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
╗
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
f
00
11
22
33
44
55
&6
'7
.8
/9"
trackable_list_wrapper
f
00
11
22
33
44
55
&6
'7
.8
/9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ь
;trace_0
<trace_1
=trace_2
>trace_32ѓ
.__inference_sequential_7_layer_call_fn_1467240
.__inference_sequential_7_layer_call_fn_1467809
.__inference_sequential_7_layer_call_fn_1467834
.__inference_sequential_7_layer_call_fn_1467695┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z;trace_0z<trace_1z=trace_2z>trace_3
┘
?trace_0
@trace_1
Atrace_2
Btrace_32Ь
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468131
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468428
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467723
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467751┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z?trace_0z@trace_1zAtrace_2zBtrace_3
МBл
"__inference__wrapped_model_1466170lstm_14_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Џ
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_rate&mџ'mЏ.mю/mЮ0mъ1mЪ2mа3mА4mб5mБ&vц'vЦ.vд/vД0vе1vЕ2vф3vФ4vг5vГ"
	optimizer
,
Hserving_default"
signature_map
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ь
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32Ѓ
)__inference_lstm_14_layer_call_fn_1468439
)__inference_lstm_14_layer_call_fn_1468450
)__inference_lstm_14_layer_call_fn_1468461
)__inference_lstm_14_layer_call_fn_1468472н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
┌
Strace_0
Ttrace_1
Utrace_2
Vtrace_32№
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468615
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468758
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468901
D__inference_lstm_14_layer_call_and_return_conditional_losses_1469044н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
"
_generic_user_object
Э
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator
^
state_size

0kernel
1recurrent_kernel
2bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

_states
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ь
etrace_0
ftrace_1
gtrace_2
htrace_32Ѓ
)__inference_lstm_15_layer_call_fn_1469055
)__inference_lstm_15_layer_call_fn_1469066
)__inference_lstm_15_layer_call_fn_1469077
)__inference_lstm_15_layer_call_fn_1469088н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zetrace_0zftrace_1zgtrace_2zhtrace_3
┌
itrace_0
jtrace_1
ktrace_2
ltrace_32№
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469233
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469378
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469523
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469668н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zitrace_0zjtrace_1zktrace_2zltrace_3
"
_generic_user_object
Э
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator
t
state_size

3kernel
4recurrent_kernel
5bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ь
ztrace_02Л
*__inference_dense_14_layer_call_fn_1469677б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zztrace_0
Ѕ
{trace_02В
E__inference_dense_14_layer_call_and_return_conditional_losses_1469688б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z{trace_0
!:d2dense_14/kernel
:2dense_14/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
«
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
ђlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
­
Ђtrace_02Л
*__inference_dense_15_layer_call_fn_1469697б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
І
ѓtrace_02В
E__inference_dense_15_layer_call_and_return_conditional_losses_1469707б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѓtrace_0
!:2dense_15/kernel
:2dense_15/bias
.:,	љ2lstm_14/lstm_cell_18/kernel
9:7
цљ2%lstm_14/lstm_cell_18/recurrent_kernel
(:&љ2lstm_14/lstm_cell_18/bias
/:-
цљ2lstm_15/lstm_cell_19/kernel
8:6	dљ2%lstm_15/lstm_cell_19/recurrent_kernel
(:&љ2lstm_15/lstm_cell_19/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
(
Ѓ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЃ
.__inference_sequential_7_layer_call_fn_1467240lstm_14_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
.__inference_sequential_7_layer_call_fn_1467809inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
.__inference_sequential_7_layer_call_fn_1467834inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
.__inference_sequential_7_layer_call_fn_1467695lstm_14_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468131inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468428inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467723lstm_14_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467751lstm_14_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
мB¤
%__inference_signature_wrapper_1467784lstm_14_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
)__inference_lstm_14_layer_call_fn_1468439inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
)__inference_lstm_14_layer_call_fn_1468450inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_14_layer_call_fn_1468461inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_14_layer_call_fn_1468472inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468615inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468758inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468901inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_14_layer_call_and_return_conditional_losses_1469044inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
█
Ѕtrace_0
іtrace_12а
.__inference_lstm_cell_18_layer_call_fn_1469724
.__inference_lstm_cell_18_layer_call_fn_1469741й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЅtrace_0zіtrace_1
Љ
Іtrace_0
їtrace_12о
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469773
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469805й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІtrace_0zїtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
)__inference_lstm_15_layer_call_fn_1469055inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
)__inference_lstm_15_layer_call_fn_1469066inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_15_layer_call_fn_1469077inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_15_layer_call_fn_1469088inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469233inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469378inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469523inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469668inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
█
њtrace_0
Њtrace_12а
.__inference_lstm_cell_19_layer_call_fn_1469822
.__inference_lstm_cell_19_layer_call_fn_1469839й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zњtrace_0zЊtrace_1
Љ
ћtrace_0
Ћtrace_12о
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469871
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469903й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zћtrace_0zЋtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_dense_14_layer_call_fn_1469677inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_dense_14_layer_call_and_return_conditional_losses_1469688inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_dense_15_layer_call_fn_1469697inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_dense_15_layer_call_and_return_conditional_losses_1469707inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
ќ	variables
Ќ	keras_api

ўtotal

Ўcount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
.__inference_lstm_cell_18_layer_call_fn_1469724inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
.__inference_lstm_cell_18_layer_call_fn_1469741inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469773inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469805inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
.__inference_lstm_cell_19_layer_call_fn_1469822inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
.__inference_lstm_cell_19_layer_call_fn_1469839inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469871inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469903inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ў0
Ў1"
trackable_list_wrapper
.
ќ	variables"
_generic_user_object
:  (2total
:  (2count
&:$d2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
&:$2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
3:1	љ2"Adam/lstm_14/lstm_cell_18/kernel/m
>:<
цљ2,Adam/lstm_14/lstm_cell_18/recurrent_kernel/m
-:+љ2 Adam/lstm_14/lstm_cell_18/bias/m
4:2
цљ2"Adam/lstm_15/lstm_cell_19/kernel/m
=:;	dљ2,Adam/lstm_15/lstm_cell_19/recurrent_kernel/m
-:+љ2 Adam/lstm_15/lstm_cell_19/bias/m
&:$d2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
&:$2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
3:1	љ2"Adam/lstm_14/lstm_cell_18/kernel/v
>:<
цљ2,Adam/lstm_14/lstm_cell_18/recurrent_kernel/v
-:+љ2 Adam/lstm_14/lstm_cell_18/bias/v
4:2
цљ2"Adam/lstm_15/lstm_cell_19/kernel/v
=:;	dљ2,Adam/lstm_15/lstm_cell_19/recurrent_kernel/v
-:+љ2 Adam/lstm_15/lstm_cell_19/bias/vБ
"__inference__wrapped_model_1466170}
012345&'./:б7
0б-
+і(
lstm_14_input         
ф "3ф0
.
dense_15"і
dense_15         г
E__inference_dense_14_layer_call_and_return_conditional_losses_1469688c&'/б,
%б"
 і
inputs         d
ф ",б)
"і
tensor_0         
џ є
*__inference_dense_14_layer_call_fn_1469677X&'/б,
%б"
 і
inputs         d
ф "!і
unknown         г
E__inference_dense_15_layer_call_and_return_conditional_losses_1469707c.//б,
%б"
 і
inputs         
ф ",б)
"і
tensor_0         
џ є
*__inference_dense_15_layer_call_fn_1469697X.//б,
%б"
 і
inputs         
ф "!і
unknown         █
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468615њ012OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ":б7
0і-
tensor_0                  ц
џ █
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468758њ012OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ":б7
0і-
tensor_0                  ц
џ ┴
D__inference_lstm_14_layer_call_and_return_conditional_losses_1468901y012?б<
5б2
$і!
inputs         

 
p 

 
ф "1б.
'і$
tensor_0         ц
џ ┴
D__inference_lstm_14_layer_call_and_return_conditional_losses_1469044y012?б<
5б2
$і!
inputs         

 
p

 
ф "1б.
'і$
tensor_0         ц
џ х
)__inference_lstm_14_layer_call_fn_1468439Є012OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "/і,
unknown                  цх
)__inference_lstm_14_layer_call_fn_1468450Є012OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "/і,
unknown                  цЏ
)__inference_lstm_14_layer_call_fn_1468461n012?б<
5б2
$і!
inputs         

 
p 

 
ф "&і#
unknown         цЏ
)__inference_lstm_14_layer_call_fn_1468472n012?б<
5б2
$і!
inputs         

 
p

 
ф "&і#
unknown         ц╬
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469233Ё345PбM
FбC
5џ2
0і-
inputs_0                  ц

 
p 

 
ф ",б)
"і
tensor_0         d
џ ╬
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469378Ё345PбM
FбC
5џ2
0і-
inputs_0                  ц

 
p

 
ф ",б)
"і
tensor_0         d
џ й
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469523u345@б=
6б3
%і"
inputs         ц

 
p 

 
ф ",б)
"і
tensor_0         d
џ й
D__inference_lstm_15_layer_call_and_return_conditional_losses_1469668u345@б=
6б3
%і"
inputs         ц

 
p

 
ф ",б)
"і
tensor_0         d
џ Д
)__inference_lstm_15_layer_call_fn_1469055z345PбM
FбC
5џ2
0і-
inputs_0                  ц

 
p 

 
ф "!і
unknown         dД
)__inference_lstm_15_layer_call_fn_1469066z345PбM
FбC
5џ2
0і-
inputs_0                  ц

 
p

 
ф "!і
unknown         dЌ
)__inference_lstm_15_layer_call_fn_1469077j345@б=
6б3
%і"
inputs         ц

 
p 

 
ф "!і
unknown         dЌ
)__inference_lstm_15_layer_call_fn_1469088j345@б=
6б3
%і"
inputs         ц

 
p

 
ф "!і
unknown         dУ
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469773џ012ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ц
#і 
states_1         ц
p 
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ц
UџR
'і$
tensor_0_1_0         ц
'і$
tensor_0_1_1         ц
џ У
I__inference_lstm_cell_18_layer_call_and_return_conditional_losses_1469805џ012ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ц
#і 
states_1         ц
p
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ц
UџR
'і$
tensor_0_1_0         ц
'і$
tensor_0_1_1         ц
џ ║
.__inference_lstm_cell_18_layer_call_fn_1469724Є012ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ц
#і 
states_1         ц
p 
ф "{бx
#і 
tensor_0         ц
QџN
%і"

tensor_1_0         ц
%і"

tensor_1_1         ц║
.__inference_lstm_cell_18_layer_call_fn_1469741Є012ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ц
#і 
states_1         ц
p
ф "{бx
#і 
tensor_0         ц
QџN
%і"

tensor_1_0         ц
%і"

tensor_1_1         цс
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469871Ћ345Ђб~
wбt
!і
inputs         ц
KбH
"і
states_0         d
"і
states_1         d
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         d
SџP
&і#
tensor_0_1_0         d
&і#
tensor_0_1_1         d
џ с
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1469903Ћ345Ђб~
wбt
!і
inputs         ц
KбH
"і
states_0         d
"і
states_1         d
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         d
SџP
&і#
tensor_0_1_0         d
&і#
tensor_0_1_1         d
џ Х
.__inference_lstm_cell_19_layer_call_fn_1469822Ѓ345Ђб~
wбt
!і
inputs         ц
KбH
"і
states_0         d
"і
states_1         d
p 
ф "xбu
"і
tensor_0         d
OџL
$і!

tensor_1_0         d
$і!

tensor_1_1         dХ
.__inference_lstm_cell_19_layer_call_fn_1469839Ѓ345Ђб~
wбt
!і
inputs         ц
KбH
"і
states_0         d
"і
states_1         d
p
ф "xбu
"і
tensor_0         d
OџL
$і!

tensor_1_0         d
$і!

tensor_1_1         d╦
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467723~
012345&'./Bб?
8б5
+і(
lstm_14_input         
p 

 
ф ",б)
"і
tensor_0         
џ ╦
I__inference_sequential_7_layer_call_and_return_conditional_losses_1467751~
012345&'./Bб?
8б5
+і(
lstm_14_input         
p

 
ф ",б)
"і
tensor_0         
џ ─
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468131w
012345&'./;б8
1б.
$і!
inputs         
p 

 
ф ",б)
"і
tensor_0         
џ ─
I__inference_sequential_7_layer_call_and_return_conditional_losses_1468428w
012345&'./;б8
1б.
$і!
inputs         
p

 
ф ",б)
"і
tensor_0         
џ Ц
.__inference_sequential_7_layer_call_fn_1467240s
012345&'./Bб?
8б5
+і(
lstm_14_input         
p 

 
ф "!і
unknown         Ц
.__inference_sequential_7_layer_call_fn_1467695s
012345&'./Bб?
8б5
+і(
lstm_14_input         
p

 
ф "!і
unknown         ъ
.__inference_sequential_7_layer_call_fn_1467809l
012345&'./;б8
1б.
$і!
inputs         
p 

 
ф "!і
unknown         ъ
.__inference_sequential_7_layer_call_fn_1467834l
012345&'./;б8
1б.
$і!
inputs         
p

 
ф "!і
unknown         И
%__inference_signature_wrapper_1467784ј
012345&'./KбH
б 
Aф>
<
lstm_14_input+і(
lstm_14_input         "3ф0
.
dense_15"і
dense_15         