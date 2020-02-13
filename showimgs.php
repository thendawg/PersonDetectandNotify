
<!DOCTYPE html>
<html>
<head>
<style>
body {
    background-color: #1f1f1f;
    color: #d8d9da;
    font-family: Rubik,sans-serif;
    font-size: 14px;
}
/* unvisited link */
a:link {
  color: #d8d9da;
}

/* visited link */
a:visited {
  color: #d8d9da;
}

/* mouse over link */
a:hover {
  color: #8892A2;
}

/* selected link */
a:active {
  color: #d8d9da;
}
a:link {
  text-decoration: none;
}

a:visited {
  text-decoration: none;
}

a:hover {
  text-decoration: none;
}

a:active {
  text-decoration: none;
}
</style>
</head>
<body>
<center>
<h3>Image Display</h3>
<?php
$images = glob("*.jpg");

foreach($images as $image) {
    echo '<img src="'.$image.'" /><br />';
}
?>
</center>
</body>
</html>
