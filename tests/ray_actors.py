import tree


x = {"bob": {
    "age": 2,
    "sex": "male"
},
"alex": {
    "age": 2,
    "sex": "female"
}
}

next_x = {"bob": {
    "age": 6,
    "sex": "male"
},
"alex": {
    "age": 6,
    "sex": "female"
}
}

print(tree.map_structure(
    lambda v, w:  v + w,
    x, next_x
))
