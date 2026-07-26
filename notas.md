Estaba pensando en algunas mejoras, definir ciertos atributos entre los actuadores

Si es controlable, y si se puede usar, pensando por ejemplo en, motor solido de cohete, no es controlable, y una vez se use, no se puede volver a usar, o que si llega un momento en el que más adelante pongo algo de manejo de combustible, si un thruster de RCS que es controlable, se queda sin combustible

Que se calcule momentos internamente

Que haya la posibilidad de que el usuario reescriba el momento generado, ejemplo (en el thurster, no defino la ubicacion (por defecto en el cg), pero defino el valor del torque nominal, y en ves de hacer el producto cruz, se calcula como la fuerza)

Output, siempre devuelve fuerza, torque (no sé si como un arreglo de numpu, un dataclass)

