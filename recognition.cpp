#include <pcl/io/pcd_io.h>
#include <pcl/io/librealsense_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/passthrough.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <chrono>
#include <boost/format.hpp>


#define FPS_CALC_BEGIN                          \
    static double duration = 0;                 \
    double start_time = pcl::getTime ();        \

#define FPS_CALC_END(_WHAT_)                    \
  {                                             \
    double end_time = pcl::getTime ();          \
    static unsigned count = 0;                  \
    if (++count == 10)                          \
    {                                           \
      std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(duration) << " Hz" <<  std::endl; \
      count = 0;                                                        \
      duration = 0.0;                                                   \
    }                                           \
    else                                        \
    {                                           \
      duration += end_time - start_time;        \
    }                                           \
  }

//using namespace pcl::tracking;

template <typename PointType>
class LibRealSenseRecognition
{
  public:
    typedef pcl::PointXYZRGBA RefPointType;

    typedef pcl::Normal NormalType;
    typedef pcl::ReferenceFrame RFType;
    typedef pcl::SHOT352 DescriptorType;

    typedef pcl::PointCloud<PointType> Cloud;
    typedef pcl::PointCloud<RefPointType> RefCloud;
    typedef typename RefCloud::Ptr RefCloudPtr;
    typedef typename RefCloud::ConstPtr RefCloudConstPtr;
    typedef typename Cloud::Ptr CloudPtr;
    typedef typename Cloud::ConstPtr CloudConstPtr;

    LibRealSenseRecognition (std::string device_id, std::string model_filename, bool show_keypoints,
    	                     bool show_correspondences, bool use_cloud_resolution, bool use_hough,
    	                     float model_ss, float scene_ss, float rf_rad, float descr_rad,
    	                     float cg_size, float cg_thresh, bool use_fixed_scene, std::string scene_filename)
    : viewer_ ("PCL LibRealSense Recognition Viewer")
    , model_filename_ (model_filename)
    , device_id_ (device_id)
    , new_cloud_ (false)
    , show_keypoints_ (show_keypoints)
    , show_correspondences_ (show_correspondences)
    , use_cloud_resolution_ (use_cloud_resolution)
    , use_hough_ (use_hough)
    , model_ss_ (model_ss)
    , scene_ss_ (scene_ss)
    , rf_rad_ (rf_rad)
    , descr_rad_ (descr_rad)
    , cg_size_ (cg_size)
    , cg_thresh_ (cg_thresh)
    , use_fixed_scene_ (use_fixed_scene)
    , scene_filename_ (scene_filename)
    {
      //
      //  Load model clouds
      //
      model_.reset (new Cloud);
      if (pcl::io::loadPCDFile (model_filename_, *model_) < 0)
      {
        std::cout << "Error loading model cloud." << std::endl;
      }

      if (use_fixed_scene_)
      {
      	scene_.reset (new Cloud);
        if (pcl::io::loadPCDFile (scene_filename_, *scene_) < 0)
        {
          std::cout << "Error loading scene cloud." << std::endl;
        }
      }
      //
      //  Set up resolution invariance
      //
      if (use_cloud_resolution_)
      {
        float resolution = static_cast<float> (computeCloudResolution (model_));
        if (resolution != 0.0f)
        {
          model_ss_   *= resolution;
          scene_ss_   *= resolution;
          rf_rad_     *= resolution;
          descr_rad_  *= resolution;
          cg_size_    *= resolution;
        }

        std::cout << "Model resolution:       " << resolution << std::endl;
        std::cout << "Model sampling size:    " << model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
      }
      
      //  Compute Model Normals
      model_normals_.reset (new pcl::PointCloud<NormalType>);
      normalEstimation (model_, *model_normals_);
     
      //  Downsample Clouds to Extract Model keypoints
      model_keypoints_.reset (new Cloud);
      extractKeyPoint (model_, *model_keypoints_, model_ss_);
      std::cout << "Model total points: " << model_->size () << "; Selected Keypoints: " << model_keypoints_->size () << std::endl;

      //  Compute Model Descriptor for Model keypoints
      model_descriptors_.reset (new pcl::PointCloud<DescriptorType>);
      computeDescriptor (model_, model_keypoints_, model_normals_, *model_descriptors_);
    }

    //Return back averaging distance between each point and its nearest neighbor 
    double
    computeCloudResolution (const CloudConstPtr &cloud)
    {
      double res = 0.0;
      int n_points = 0;
      int nres;
      std::vector<int> indices (2);
      std::vector<float> sqr_distances (2);
      pcl::search::KdTree<PointType> tree;
      tree.setInputCloud (cloud);

      for (size_t i = 0; i < cloud->size (); ++i)
      {
        if (! pcl_isfinite ((*cloud)[i].x))
        {
          continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
          res += sqrt (sqr_distances[1]);
          ++n_points;
        }
       }
      if (n_points != 0)
      {
        res /= n_points;
      }
      return res;
    }

    void
    normalEstimation (const CloudConstPtr& cloud,
    	              pcl::PointCloud<pcl::Normal>& result)
    {
      FPS_CALC_BEGIN;
      pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
      norm_est.setKSearch (10);//10
      norm_est.setInputCloud (cloud);
      norm_est.compute (result);
      FPS_CALC_END("normalEstimation");
    }

    void
    extractKeyPoint (const CloudConstPtr& cloud,
    	             Cloud& result,
    	             float ss)
    {
      FPS_CALC_BEGIN;
      pcl::UniformSampling<PointType> uniform_sampling;
      uniform_sampling.setInputCloud (cloud);
      uniform_sampling.setRadiusSearch (ss);
      uniform_sampling.filter (result);
      FPS_CALC_END("extractKeyPoint");	
    }

    void
    computeDescriptor (const CloudConstPtr& cloud,
    	               CloudPtr& keypoints,
    	               pcl::PointCloud<pcl::Normal>::Ptr& normals,
    	               pcl::PointCloud<DescriptorType>& descriptor)
    {
      FPS_CALC_BEGIN;
      pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
      descr_est.setRadiusSearch (descr_rad_);
      descr_est.setInputCloud (keypoints);
      descr_est.setInputNormals (normals);
      descr_est.setSearchSurface (cloud);
      descr_est.compute (descriptor);
      FPS_CALC_END("computeDescriptor");
    }

    void 
    filterPassThrough (const CloudConstPtr &cloud, Cloud &result)
    {
      FPS_CALC_BEGIN;
      pcl::PassThrough<PointType> pass;
      //CloudPtr res (new pcl::PointCloud<pcl::PointXYZRGBA>);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (-3.0, 0.0);//changed
      pass.setKeepOrganized (false);
      pass.setInputCloud (cloud);
      pass.filter (result);
      FPS_CALC_END("filterPassThrough");
    }

    void
    viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
      boost::mutex::scoped_lock lock (mtx_);

      if (!cloud_pass_)
      {
      	boost::this_thread::sleep (boost::posix_time::seconds (1));
      	return;// ?
      }

      if (new_cloud_)
      {
      	CloudPtr cloud_pass;
      	cloud_pass = cloud_pass_;
      	if (!viz.updatePointCloud (cloud_pass, "cloudpass"))
      	{
      	  viz.addPointCloud (cloud_pass, "cloudpass");
      	  viz.resetCameraViewpoint ("cloudpass");
      	}

        RefCloudPtr off_scene_model (new RefCloud ());
        RefCloudPtr off_scene_model_keypoints (new RefCloud ());

        if (show_correspondences_ || show_keypoints_)
        {
          //  We are translating the model so that it doesn't end in the middle of the scene representation
          pcl::transformPointCloud (*model_, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
          pcl::transformPointCloud (*model_keypoints_, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

          pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
          if (!viz.updatePointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model"))
            viz.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
        }

        if (show_keypoints_)
        {
          std::cout << "Scene Keypoints_ size: " << scene_keypoints_->size () << std::endl;	
          pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints_, 0, 0, 255);
          if (!viz.updatePointCloud (scene_keypoints_, scene_keypoints_color_handler, "scene_keypoints"))
            viz.addPointCloud (scene_keypoints_, scene_keypoints_color_handler, "scene_keypoints");
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

          pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
          if (!viz.updatePointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints"))  
            viz.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
        }

        for (size_t i = 0; i < rototranslations_.size (); ++i)
        {
          RefCloudPtr rotated_model (new RefCloud ());
          pcl::transformPointCloud (*model_, *rotated_model, rototranslations_[i]);

          std::stringstream ss_cloud;
          ss_cloud << "instance" << i;

          pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
          if (!viz.updatePointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ()))
            viz.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
        }
      }
      new_cloud_ = false;
    }

    void
    cloud_cb (const CloudConstPtr &cloud)
    {
      boost::mutex::scoped_lock lock (mtx_);
      //CloudPtr scene (new Cloud ());
      CloudPtr scene_keypoints (new Cloud ());
      pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
      pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
      cloud_pass_.reset (new Cloud);
      //std::cout << "scene: " << scene_->size () << std::endl;
      if (use_fixed_scene_)
      {
        cloud_pass_ = scene_;
      }
      else
      {
        filterPassThrough (cloud, *cloud_pass_);
      }
      std::cout << "cloud_pass: " << cloud_pass_->size () << std::endl;
      normalEstimation (cloud_pass_, *scene_normals);
      
      extractKeyPoint (cloud_pass_, *scene_keypoints, scene_ss_);
      float time = 0;
      auto t0 = std::chrono::high_resolution_clock::now ();
      computeDescriptor (cloud_pass_, scene_keypoints, scene_normals, *scene_descriptors);
      std::cout << "Scene Keypoints size: " << scene_keypoints->size () << std::endl;
      std::cout << "Initial process done" << std::endl;
      auto t1 = std::chrono::high_resolution_clock::now ();
      time = std::chrono::duration<float> (t1 - t0).count ();
      printf("Load clouds time duration is %.4f\n",time);
      t0=t1;
      //
      //  Find Model-Scene Correspondences with KdTree
      //
      pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

      pcl::KdTreeFLANN<DescriptorType> match_search;
      match_search.setInputCloud (model_descriptors_);
      //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
      for (size_t i = 0; i < scene_descriptors->size (); ++i)
      {
      	std::vector<int> neigh_indices (1);
      	std::vector<float> neigh_sqr_dists (1);
      	if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
        {
          continue;
        }
        //Searching domain is model_descr, point is scene)descr point, only find the nearest point, store its indice and dist
        int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
          //corr between model indice and scene indice
          pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
          model_scene_corrs->push_back (corr);
        }
      }

      t1 = std::chrono::high_resolution_clock::now ();
      time = std::chrono::duration<float> (t1 - t0).count ();
      printf("Find Model_scene time duration is %.4f\n",time);
      t0=t1;
      std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
      //
      //  Actual Clustering
      //
      //std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
      std::vector<pcl::Correspondences> clustered_corrs;
      rototranslations_.clear ();
      //  Using Hough3D
      if (use_hough_)
      {
        //
        //  Compute (Keypoints) Reference Frames only for Hough
        //
        pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
        pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

        pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
        rf_est.setFindHoles (true);
        rf_est.setRadiusSearch (rf_rad_);

        rf_est.setInputCloud (model_keypoints_);
        rf_est.setInputNormals (model_normals_);
        rf_est.setSearchSurface (model_);
        rf_est.compute (*model_rf);

        rf_est.setInputCloud (scene_keypoints);
        rf_est.setInputNormals (scene_normals);
        rf_est.setSearchSurface (cloud_pass_);
        rf_est.compute (*scene_rf);

        //  Clustering
        pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
        clusterer.setHoughBinSize (cg_size_);
        clusterer.setHoughThreshold (cg_thresh_);
        clusterer.setUseInterpolation (true);
        clusterer.setUseDistanceWeight (false);

        clusterer.setInputCloud (model_keypoints_);
        clusterer.setInputRf (model_rf);
        clusterer.setSceneCloud (scene_keypoints);
        clusterer.setSceneRf (scene_rf);
        clusterer.setModelSceneCorrespondences (model_scene_corrs);

        clusterer.recognize (rototranslations_, clustered_corrs);
      }
      else //  Using GeometricConsistency
      {
        pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
        gc_clusterer.setGCSize (cg_size_);
        gc_clusterer.setGCThreshold (cg_thresh_);

        gc_clusterer.setInputCloud (model_keypoints_);
        gc_clusterer.setSceneCloud (scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

        gc_clusterer.recognize (rototranslations_, clustered_corrs);
      }
      //
      //  Output results
      //
      std::cout << "Model instances found: " << rototranslations_.size () << std::endl;
      for (size_t i = 0; i < rototranslations_.size (); ++i)
      {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations_[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations_[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
      }
      scene_keypoints_ = scene_keypoints;
      new_cloud_ = true;
    }

    void
    run ()
    {
      pcl::Grabber* interface = new pcl::LibRealSenseGrabber (device_id_);
      boost::function<void (const CloudConstPtr&)> f =
        boost::bind (&LibRealSenseRecognition::cloud_cb, this, _1);
      interface->registerCallback (f);
      
      viewer_.runOnVisualizationThread (boost::bind(&LibRealSenseRecognition::viz_cb, this, _1), "viz_cb");

      interface->start ();

      while (!viewer_.wasStopped ())
        boost::this_thread::sleep(boost::posix_time::seconds(1));
      interface->stop ();
    }

    CloudPtr model_;
    CloudPtr model_keypoints_;
    CloudPtr cloud_pass_;
    CloudPtr scene_keypoints_;
    CloudPtr scene_;
    pcl::PointCloud<NormalType>::Ptr model_normals_;
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations_;

    pcl::visualization::CloudViewer viewer_;
    std::string device_id_;
    std::string model_filename_;
    std::string scene_filename_;

    boost::mutex mtx_;
    bool new_cloud_;
    bool show_keypoints_;
    bool show_correspondences_;
    bool use_cloud_resolution_;
    bool use_hough_;
    bool use_fixed_scene_;
    float model_ss_;
    float scene_ss_;
    float rf_rad_;
    float descr_rad_;
    float cg_size_;
    float cg_thresh_;

};

void
usage (char* argv)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*                   Recognition Tutorial - Usage Guide                    *" << std::endl;
  std::cout << "*                        for RealSense Camera Only                        *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;	
  std::cout << "usage: " << argv[0] << " <device_id> model_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "     -k:                          Show used keypoints." << std::endl;
  std::cout << "     -r:                          Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                                  each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC):      Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --scene scene_filename.pcd:  Use fixed scene pcd file for recognition" << std::endl;
  std::cout << "     --model_ss val:              Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:              Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:                Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:             Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:               Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:             Clustering threshold (default 5)" << std::endl << std::endl;
}

int
main (int argc, char** argv)
{
  bool show_keypoints = false;
  bool show_correspondences = false;
  bool use_cloud_resolution = false;
  bool use_hough = true;
  bool use_fixed_scene =false;
  float model_ss = 0.01f;
  float scene_ss = 0.03f;
  float rf_rad = 0.015f;
  float descr_rad = 0.02f;
  float cg_size = 0.01f;
  float cg_thresh = 5.0f;
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (pcl::console::find_switch (argc, argv, "-k") || pcl::console::find_switch (argc, argv, "-K"))
  {
    show_keypoints = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c") || pcl::console::find_switch (argc, argv, "-C"))
  {
    show_correspondences = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r") || pcl::console::find_switch (argc, argv, "-R"))
  {
    use_cloud_resolution = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      usage (argv[0]);
      exit (-1);
    }
  }

  std::string scene_filename;
  if (pcl::console::parse_argument (argc, argv, "--scene", scene_filename) != -1)
  {
    use_fixed_scene = true;
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh);


  if (filenames.size () < 1)
  {
    std::cout << "Filenames missing.\n";
    usage (argv[0]);
    exit (-1);
  }
  std::string model_filename = argv[filenames[0]];
  std::string device_id = std::string (argv[1]);

  //open RealSense
  LibRealSenseRecognition<pcl::PointXYZRGBA> v (device_id, model_filename, show_keypoints, show_correspondences,
                                                use_cloud_resolution, use_hough, model_ss, scene_ss,
                                                rf_rad, descr_rad, cg_size, cg_thresh, use_fixed_scene, scene_filename);
  v.run ();
}

