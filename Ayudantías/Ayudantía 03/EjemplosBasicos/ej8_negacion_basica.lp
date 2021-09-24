% Existen cinco cursos:
course(1..5).

% Los dos primeros cursos son obligatorios:
required(1).
required(2).

% Se deben inscribir cuatro cursos:
4 {selected(C): course(C)} 4.

% No es posible no inscribir los cursos obligatorios
:- not selected(C), required(C).

% Solo interesa saber qué cursos se inscribieron:
#show selected/1.
