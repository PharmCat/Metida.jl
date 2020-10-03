DATA df0 ;
input subject $ sequence $ period  $ formulation $ var;
datalines;
1 1 1 1 1.0
1 1 2 2 1.1
1 1 3 1 1.2
1 1 4 2 1.3
2 1 1 1 2.0
2 1 2 2 2.1
2 1 3 1 2.4
2 1 4 2 2.2
3 2 1 2 1.3
3 2 2 1 1.5
3 2 3 2 1.6
3 2 4 1 1.4
4 2 1 2 1.5
4 2 2 1 1.7
4 2 3 2 1.3
4 2 4 1 1.4
5 2 1 2 1.5
5 2 2 1 1.7
5 2 3 2 1.2
5 2 4 1 1.8
;

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=CSH SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=VC SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  subject/TYPE=VC G V;
RUN;

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  period/TYPE=VC G V;
RANDOM  formulation/TYPE=VC G V;
RUN;

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=UN(1) SUB=subject G V;
RUN;
