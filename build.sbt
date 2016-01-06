// Versions
val scalaVer = "2.11.7"
val sparkVer = "1.6.0"
val aolmlVer = "1.0"

// Log Level
logLevel := Level.Warn


// App Name
name := "aol-ml"
version := aolmlVer

scalaVersion := scalaVer


/*
 * External Libraries
 */
libraryDependencies ++= Seq(
  "org.apache.spark"              %%    "spark-core"    %     sparkVer,
  "org.apache.spark"              %%    "spark-mllib"   %     sparkVer,
  "org.apache.spark"              %%    "spark-sql"     %     sparkVer,
  "com.github.scopt"              %%    "scopt"         %     "3.3.0"
)

/*
 * Compilation
 */
scalacOptions in (Compile,doc) ++= Seq("-groups", "-implicits")


/*
 * Assembly
 */
assemblyOption in assembly ~= { _.copy(cacheOutput = false) }


assemblyMergeStrategy in assembly := {
  case PathList(ps @ _*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xml" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".dtd" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xsd" => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

/*
 * Unit testing
 */
