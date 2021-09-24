% Para declarar átomos lo hacemos de la siguiente manera
year(2019).
year(2020).
year(2021).

% Si queremos definir átomos que representen varios años consecutivos podemos hacerlo a mano, de-
% clarando una sentencia para cada uno de sus valores, pero también podemos automatizarlo y resumir-
% lo con la siguiente sintáxis:
year(2019..2021).

% También se pueden definir átomos en el programa para realizar lo anterior:
#const end_1 = 2021.
year_const(2019..end_1).

% También se puede hacer ingresando el valor de constantes mediante los flags en la terminal (si se 
% referencia la variable y no se ingresó su valor, entonces el programa se interrumpe):
% Para ejecutar: clingo ej1_generador_átomos.lp -c end_2=2021
year_flag(2019..end_2).
