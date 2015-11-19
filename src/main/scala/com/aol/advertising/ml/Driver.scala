/**
 * Project          : aol-ml
 * File             : Driver
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/26/15
 *
 * Descriptor       : 
 * " The class usage goes here. "
 *
 */

package com.aol.advertising.ml

import scala.collection.immutable.HashSet

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}

import org.apache.spark.ml.classification.{FactorizationMachine, LogisticRegression}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}

// Scopt Input parser case class
case class ScoptConfig(locally: Boolean = false,
                       input: String = "",
                       output: String = ""
                        )


object Driver {

  def composeFilesList(path: String,
                       fileTypeExtension: String)
  : Array[String] =
    if (path.endsWith(fileTypeExtension)) // A single specific file provided
      Array(path)
    else {

      val fs = FileSystem.get(new Configuration())

      val fileStatus = fs.globStatus(new Path(path))
      if (fileStatus != null) {
        fileStatus.map(f => {
          f.getPath.toString
        }).filter(_.endsWith(fileTypeExtension))
      }
      else
        throw new RuntimeException("No input file of type '" +
          fileTypeExtension + "' found at the location specified by '" +
          path)
    }


  def main(args: Array[String]): Unit = {

    /*
     *  The commandline argument handler
     */
    val parser = new scopt.OptionParser[ScoptConfig]("(sbt) run") {
      head("Commandline Arguments", "0.x")

      opt[Unit]('l', "runlocally")
        .action { (_, c) => c.copy(locally = true) }
        .text("run locally instead of on Spark cluster/client mode")

      opt[String]('i', "input")
        .required()
        .action { (x, c) => c.copy(input = x) }
        .text("input (file path) is a required property [type: String]")

      opt[String]('o', "output")
        .required()
        .action { (x, c) => c.copy(output = x) }
        .text("output (file path) is a required property [type: String]")

      help("help")
        .text("prints this usage text")
    }
    // Parse the input arguments
    val clArgs = parser.parse(args, ScoptConfig()) match {
      case Some(conf) => conf
      case _ => throw new RuntimeException("Error unrolling commandline arguments. Run 'sbt run --help'.")
    }


    /*
     * Compose input file list to be processed
     */
    val files: Array[String] = composeFilesList(clArgs.input, ".gz")

    if (files.length < 1)
      throw new RuntimeException("No file was found.")

    if (files.length > 1)
      throw new RuntimeException("Currently only one file at a time is served.")


    val trainingDataFile =
      if ( MLUtil.FileLoader.isFound(files.head) )
        files.head
      else
        throw new RuntimeException("Datafile '" + files.head + "' not found.")

    /*
     * Spark Context
     */
    val sparkConf = new SparkConf()
      .setAppName("FactorizationMachine-demo")
    if (clArgs.locally)
      sparkConf.setMaster("local[4]")

    val sc = new SparkContext(sparkConf)


    val sqlContext = SQLContext.getOrCreate(sc)
    import sqlContext.implicits._

    import MLUtil.FileLoader._



    /*
     * Features set
     */
    val columnsToSelect = HashSet(
      "isclk",
      "ad_ctr",
      "advertiser_ctr",
      "campaign_ctr",
      "site_ad_ctr",
      "site_campaign_ctr",
      "site_ctr",
      "adid",
      "browser_5",
      "cc",
      "user_cookie_ability_1604",
      "user_acid_age_10",
      "user_network_clicker_546",
      "user_network_actor_547"
    )

    /*
     * Categorical features/variables
     */
    val categoricalVariables = HashSet(
      "adid",
      "browser_5",
      "cc",
      "user_cookie_ability_1604",
      "user_acid_age_10",
      "user_network_clicker_546",
      "user_network_actor_547"
    )

    /*
     * The training data
     * (and the feature index map)
     */
    val trainingDataRddAndFeatureMap = sc.loadCsvFileAsRDD(
      trainingDataFile,
      withHeader = true,
      separator = " ")
     // selectColumns = columnsToSelect,
     // categoricalFeatures = categoricalVariables)

    val trainingData = trainingDataRddAndFeatureMap.data
      .repartition(sc.defaultMinPartitions * 3)
      .map(point => LabeledPoint( if (point.label == -1.0) 0.0 else point.label, point.features))
      .toDF("label", "features")


    trainingData.show(20, truncate = false)


    /*
    * Test/validation data
    * (with the same loader specifications
    * as used in importing training data.)
    */
    val testData = sc.loadCsvFileAsRDD(
      trainingDataFile.replaceFirst(".TR.gz", ".TR.gz"),
      withFeatureIndexing = trainingDataRddAndFeatureMap.featureIndexMap,
      withLoaderParams = trainingDataRddAndFeatureMap.loaderParams
    ).data
      .repartition(sc.defaultMinPartitions * 3)
      .map(point => LabeledPoint( if (point.label == -1.0) 0.0 else point.label, point.features))
      .toDF("label", "features")


    /*
     * Logistic Regression model
     */
    val lr = new LogisticRegression()
      .setMaxIter(50)
      .setRegParam(0.000)
      .setElasticNetParam(0.95)

    /*
     * Factorization Machine
     */
    val fMachine = new FactorizationMachine(1)
      .setMaxIter(1000)
      .setRegParam(0.000)
      .setElasticNetParam(0.95)

    /*
     * Fit models
     */
    val lrModel = lr.fit(trainingData)
    val fmModel = fMachine.fit(trainingData)


    println(s"${lrModel.intercept} ${lrModel.weights}")
    println(s"${fmModel.coefficients.intercept} ${fmModel.coefficients.linear} ${fmModel.coefficients.quadratic}")

//    /*
//     * Test data matching
//     */
//    val lrTest = lrModel
//      .transform(testData)
//      .select("label","probability", "prediction")
//      .map{
//        case Row(label: Double, probability: Vector, prediction: Double) => (probability(1), label)
//      }
//      .zipWithIndex()
//      .map{ case ((probability, label), index) => (probability, label, index)}
//      .toDF("lrProbability", "lrLabel", "id")
//
//
//    val fmTest = fmModel
//      .transform(testData)
//      .select("label", "probability", "prediction")
//      .map{
//        case Row(label: Double, probability: Vector, prediction: Double) => (probability(1), label)
//      }
//      .zipWithIndex()
//      .map{ case ((probability, label), index) => (probability, label, index)}
//      .toDF("fmProbability", "fmLabel", "id")
//
//
//    val joinedDF = fmTest
//      .join(lrTest, usingColumn = "id")
//      //.join(newlrTest, usingColumn = "id")
//
//    joinedDF
//    //  .filter(joinedDF("fmProbability") !== joinedDF("lrProbability"))
//      .show(truncate = false)

    sc.stop()
  }
}
