#include <cstdio>
#include <cmath>
#include <iostream>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

static void Help()
{
    printf("\nRun whis 2 arguments:\n"
        "\tFirst: \n"
        "\t\tw - Wave algorithm\n"
        "\t\tm - Morphology (very slow)\n"
        "\tSecond: \n"
        "\t\tPath to image\n\n");
}

static void HelpWave()
{
    printf("\nLeft click to show start and finish points\n\n");
}

///////////////////////////////////////////////////////////////////////////////

static void onMouse (int event, int x, int y, int, void* ptr);
void Morphology_Operation (int operation, int dilation_elem, int dilation_size, Mat &src, Mat &dilation_dst);

///////////////////////////////////////////////////////////////////////////////

class Labyrinth
{
    public:
        Labyrinth ();
        virtual ~Labyrinth () {};

        void Init (Mat &input);

        Point get_start ()  const;
        Point get_finish () const;
        void set_start (Point start);
        void set_finish (Point finish);
        void set_image (Mat &input);

        virtual int FindWay () = 0;

        Mat output_;

    protected:
        Mat   input_;
        Point start_, finish_;
};

class LabyrinthWave: public Labyrinth
{
    public:
        ~LabyrinthWave ();

        void Start (Mat &input);
        int FindWay ();

    private:
        void ExtractAndDrawWay (Point index);
        bool Wave (Point currentPoint, Point direction);

        int* visits_;                             /// the coordinates of where you came from
        std::queue  <Point> queueOfVertex_;
};

class LabyrinthMorphology: public Labyrinth
{
    public:
        void Start (Mat &input);
        int FindWay ();

    private:
        bool IsEqual (const Mat &image1, const Mat &image2);
};

///////////////////////////////////////////////////////////////////////////////

Labyrinth::Labyrinth ()
{
    start_  = Point (-1, -1);
    finish_ = Point (-1, -1);
}

void Labyrinth::Init (Mat &input)
{
    set_image (input);

    Mat mask;
    Morphology_Operation (MORPH_OPEN, 0, 10, output_, mask);

    /// Extract the labyrinth. Remove the white borders
    for (int i = 0; i < output_.rows; ++i)
    {
        for (int j = 0; j < output_.cols; ++j)
        {
            if (mask.at <Vec3b> (i, j) != Vec3b (0, 0, 0))
            {
                input_.at  <Vec3b> (i, j) = Vec3b (240, 240, 240);
                output_.at <Vec3b> (i, j) = Vec3b (210, 210, 210);
            }
        }
    }
}

Point Labyrinth::get_start   () const     {   return start_;}
Point Labyrinth::get_finish  () const     {   return finish_;}
void Labyrinth::set_start  (Point start)  {   start_  = start; }
void Labyrinth::set_finish (Point finish) {   finish_ = finish;}

void Labyrinth::set_image (Mat &input)
{
    input.convertTo (input_, CV_8UC3);
    input.convertTo (output_, CV_8UC3);
}

///////////////////////////////////////////////////////////////////////////////

LabyrinthWave::~LabyrinthWave ()
{
    delete [] visits_;
}

void LabyrinthWave::Start (Mat &input)
{
    HelpWave ();

    Labyrinth::Init (input);
    setMouseCallback ("Result", onMouse, (void*) this);

    visits_ = new int [input_.rows * input_.cols];

    for (int i = 0, lim = input_.rows * input_.cols; i < lim; ++i)
        visits_ [i] = -1;
}

void LabyrinthWave::ExtractAndDrawWay (Point index)
{
    while (index != Point (0, 0))
    {
        output_.at <Vec3b> (index.x, index.y) = Vec3b (255, 0, 0);

        int x = index.x,
            y = index.y;

        index.x = visits_ [x * output_.cols + y] / output_.cols;
        index.y = visits_ [x * output_.cols + y] % output_.cols;
    }
}

/// Return 0, if finish not found, or 1 otherwise
bool LabyrinthWave::Wave (Point currentPoint, Point direction)
{
    Point newPoint = currentPoint + direction;

    /// Check for array limits
    if ((newPoint.x >= 0) && (newPoint.x < input_.rows) &&
        (newPoint.y >= 0) && (newPoint.y < input_.cols))
    {
        int index_new  = newPoint.x * input_.cols + newPoint.y;
        int index_from = currentPoint.x * input_.cols + currentPoint.y;

        /// If we here in the first timetouch README.md
        if (visits_ [index_new] == -1)
        {
            if (newPoint == finish_)
            {
                visits_ [index_new] = index_from;
                ExtractAndDrawWay (newPoint);
                return 1;
            }

            /// If we can move through this place. (Small change color)
            Vec3b diff =    input_.at <Vec3b> (newPoint.x, newPoint.y) -
                            input_.at <Vec3b> (currentPoint.x, currentPoint.y);

            if (abs (diff[0]) < 2)
            {
                queueOfVertex_.push (newPoint);
                visits_ [index_new] = index_from;
            }
        }
    }

    return 0;
}

int LabyrinthWave::FindWay ()
{
    if (start_.x < 0)
    {
        printf ("Can't find start\n");
        return -1;
    }

    queueOfVertex_.push (start_);
    visits_ [start_.x * input_.cols + start_.y] = 0;

    bool isFindFinish = 0;
    while ((!queueOfVertex_.empty ()) && (!isFindFinish))
    {
        Point curr = queueOfVertex_.front ();
        queueOfVertex_.pop ();

        /// Vertical, horizontal
        isFindFinish |= Wave (curr, Point (-1,  0));
        isFindFinish |= Wave (curr, Point ( 0,  1));
        isFindFinish |= Wave (curr, Point ( 1,  0));
        isFindFinish |= Wave (curr, Point ( 0, -1));

        /// Diagonal
        isFindFinish |= Wave (curr, Point (-1, -1));
        isFindFinish |= Wave (curr, Point (-1,  1));
        isFindFinish |= Wave (curr, Point ( 1, -1));
        isFindFinish |= Wave (curr, Point ( 1,  1));
    }

    if (!isFindFinish)
    {
        printf ("Can't find finish\n");
        return -1;
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

void LabyrinthMorphology::Start (Mat &input)
{
    Labyrinth::Init (input);
    FindWay ();
}

int LabyrinthMorphology::FindWay ()
{
    Mat mask, prevState;
    input_.copyTo (mask);

    do
    {
        mask.copyTo (prevState);
        Morphology_Operation (MORPH_OPEN, 2, 4, mask, mask);
        Morphology_Operation (MORPH_OPEN, 0, 4, mask, mask);
        threshold (mask, mask, 210, 255, THRESH_BINARY);

//        imshow ("Morphology", mask);
//        waitKey (1);
    }
    while (!IsEqual (mask, prevState));

    Morphology_Operation (MORPH_ERODE, 2, 3, mask, mask);

    /// Copy the way from mask
    for (int i = 0; i < mask.rows; ++i)
    {
        for (int j = 0; j < mask.cols; ++j)
        {
            if ( (mask.at <Vec3b> (i, j) == Vec3b (255, 255, 255)) &&
                 (output_.at <Vec3b> (i, j) == Vec3b (255, 255, 255)) )
            {
                output_.at <Vec3b> (i, j) = Vec3b (255, 0, 0);
            }
        }
    }

    return 0;
}

bool LabyrinthMorphology::IsEqual (const Mat &image1, const Mat &image2)
{
    for (int i = 0; i < image1.rows; ++i)
        for (int j = 0; j < image1.cols; ++j)
            if (image1.at <Vec3b> (i, j) != image2.at <Vec3b> (i, j))
                return 0;

    return 1;
}

///////////////////////////////////////////////////////////////////////////////

static void onMouse (int event, int x, int y, int, void* ptr)
{
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    Labyrinth *labyrinth = (Labyrinth *) ptr;

    /// If a point inside the labyrinth
    if (labyrinth->output_.at <Vec3b> (Point (x, y)) == Vec3b (255, 255, 255))
    {
        /// If we haven't start point
        if (labyrinth->get_start ().x < 0)
        {
            labyrinth->set_start (Point (y, x));
            circle (labyrinth->output_, Point (x, y), 4, Scalar (0, 255, 0));
        }
        /// If we haven't finish
        else if (labyrinth->get_finish ().x < 0)
        {
            labyrinth->set_finish (Point (y, x));
            circle (labyrinth->output_, Point (x, y), 4, Scalar (0, 0, 255));

            labyrinth->FindWay ();
        }
    }

    imshow ("Result", labyrinth->output_);
}

void Morphology_Operation (int operation, int dilation_elem, int dilation_size, Mat &src, Mat &dilation_dst)
{
    int dilation_type = 0;

    if      (dilation_elem == 0)    { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1)    { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2)    { dilation_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement (dilation_type,
                                         Size (2 * dilation_size + 1, 2 * dilation_size + 1),
                                         Point (dilation_size, dilation_size));
    /// Apply the operation
    morphologyEx (src, dilation_dst, operation, element);
}

///////////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv)
{
    if (argc != 3)
    {
        Help ();
        return -1;
    }

    Mat image = imread (argv [2], 1);

    if (!image.data)
    {
        std::cerr << "Can't load " << argv [2] << " image file" << std::endl;
        return -1;
    }

    namedWindow ("Result");

    if (argv [1][0] == 'w')
    {
        LabyrinthWave labyrinth;

        labyrinth.Start (image);
        imshow ("Result", labyrinth.output_);
        waitKey ();
    }
    else if (argv [1][0] == 'm')
    {
        LabyrinthMorphology labyrinth;

        labyrinth.Start (image);
        imshow ("Result", labyrinth.output_);
        waitKey ();
    }
    else
    {
        printf ("Wrong parameters");
        Help ();
        return -1;
    }

    return 0;
}
