<!DOCTYPE html>
<html>
<head>
<script>
function timedRefresh(timeoutPeriod) {
	setTimeout("location.reload(true);",timeoutPeriod);
}

window.onload = timedRefresh(10000);
</script>
<style>
body {
    background-color: #1b1a1a;
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
<h3>Camera Detection Log</h3>
<?php
$fArray = file("imgdir/detect.log");
$len = sizeof($fArray);
for($i=$len -50;$i<$len ;$i++)
{
   echo $fArray[$i]."<br>";
}
?>
</center>
</body>
</html>
