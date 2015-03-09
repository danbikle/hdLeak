
package water.droplets

import java.io._
import org.apache.spark.h2o._
import org.apache.spark.{SparkFiles, SparkContext, SparkConf}

import water.fvec._
import water.fvec.DataFrame

import hex.deeplearning.DeepLearning
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation

object SparklingWaterDroplet {


  def main(args: Array[String]) {

    val train_count_a = Array(1001)
    val epochnum_a    = Array(20)
    val hid_a         = Array(1)
    (train_count_a).foreach(train_count => {
      println(train_count)
      (epochnum_a).foreach(epochnum => {
        println(epochnum)
        (hid_a).foreach(hid => {
          println(hid)
          main2(train_count,epochnum,hid)})})})}

  def main2(train_count_in:Int, epochnum_in:Int, hid_in:Int) {

    // Create Spark Context
    val conf = configure("aDroplet")
    val sc = new SparkContext(conf)

    // Create H2O Context
    val h2oContext = new H2OContext(sc).start()
    import h2oContext._
    
    // I should read some CSV data.
    // I actually only have about 2590 rows in wide1.csv
    val observation_count = 2590

    val pred_count = observation_count - train_count_in -2
    // val pred_count = 11

    var predictions_array = new Array[String](pred_count)
    val top_folder = "/home/dan/a18"
    val observations_rdd = sc.textFile(top_folder+"/swd/data/wide1.csv").repartition(1).cache()

    // wide1.csv columns:
    // cdate,cp,pctlead,dow,dom,moy,eem3,eem4,eem5,eem6,efa3,efa4, ...
    // C1,   C2,C3,     C4, ...

    val observations_array = observations_rdd.take(observation_count)
    // Some DL-params:
    var hid1 = 99*hid_in
    var hid2 = 199*hid_in
    var hid3 = 99*hid_in

    (0 to pred_count-1).foreach(rnum => {
      val oos_array1     = observations_array.take(rnum+1).drop(rnum)
      val training_array = observations_array.drop(rnum+1).take(rnum+train_count_in)

      val oos_writer = new PrintWriter(new File("/tmp/oos1.csv" ))
      oos_array1.foreach(elm => oos_writer.write(elm+"\n"))
      oos_writer.close

      val training_writer = new PrintWriter(new File("/tmp/train.csv" ))
      training_array.foreach(elm => {
        elm.foreach(elm2 => training_writer.write(elm2))
        training_writer.write("\n")})
      training_writer.close

      val oosf     = new java.io.File("/tmp/oos1.csv" )
      val oosDF    = new DataFrame(oosf)

      val trainf     = new java.io.File("/tmp/train.csv" )
      val trainDF    = new DataFrame(trainf)

      val dlParams    = new DeepLearningParameters()
      dlParams._train = trainDF('C3,'C4,'C5,'C7,'C8,'C9,'C10,'C11,'C12,'C13,'C14,'C15,'C16,'C17,'C18,'C19,'C20,'C21,'C22,'C23,'C24,'C25,'C26,'C27,'C28,'C29,'C30,'C31,'C32,'C33,'C34,'C35,'C36,'C37,'C38,'C39,'C40,'C41,'C42,'C43,'C44,'C45,'C46)
      dlParams._response_column = 'C3

      dlParams._epochs     = epochnum_in
      dlParams._activation = Activation.RectifierWithDropout
      dlParams._rate       = 0.03
      dlParams._hidden     = Array[Int](hid1,hid2,hid3)
      // dlParams._rate_annealing = 0.000001

      val dl      = new DeepLearning(dlParams)
      val dlModel = dl.trainModel.get

      val predDF  = dlModel.score(oosDF)('predict)

      val pred    = predDF.vec(0).at(0)
      println("Prediction is: "+pred)
      // I should match pred to oos-date
      val oos_date = oosDF('C1).vec(0).at(0)
      val oos_cp   = oosDF('C2).vec(0).at(0)
      val oos_pctlead         = oosDF('C3).vec(0).at(0)
      predictions_array(rnum) = oos_date+","+oos_cp+","+pred+","+oos_pctlead

      oosDF.delete()
      trainDF.delete()
      predDF.delete()

    }) //.foreach

    // I should write predictions to CSV
    val csvf = top_folder+"/swd/data/predDL"+pred_count+"_"+train_count_in+"_"+epochnum_in+"_"+hid1+"_"+hid2+"_"+hid3+".csv"
    val pred_writer = new PrintWriter(new File(csvf))
    predictions_array.foreach(elm => pred_writer.write(elm+"\n"))
    pred_writer.close

    sc.stop()
  }

  def configure(appName:String = "aDemo"):SparkConf = {
    val conf = new SparkConf()
      .setAppName(appName)
    conf.setIfMissing("spark.master", sys.env.getOrElse("spark.master", "local"))
    conf
  }
}
