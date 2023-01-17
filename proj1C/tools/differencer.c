#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <file1.pnm> <file2.pnm>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *filename1 = argv[1];
    char *filename2 = argv[2];

    FILE *f1 = fopen(filename1, "r");
    if (f1 == NULL)
    {
        fprintf(stderr, "Unable to open \"%s\" for reading\n", filename1);
        exit(EXIT_FAILURE);
    }
    FILE *f2 = fopen(filename2, "r");
    if (f2 == NULL)
    {
        fprintf(stderr, "Unable to open \"%s\" for reading\n", filename2);
        exit(EXIT_FAILURE);
    }

    char p6[1024];
    int  f1x, f1y, max1;
    fscanf(f1, "%s\n%d %d\n%d\n", p6, &f1x, &f1y, &max1);
    if (strcmp(p6, "P6") != 0)
    {
        fprintf(stderr, "File \"%s\" is not of type P6.\n", filename1);
        exit(EXIT_FAILURE);
    }
    if (max1 != 255)
    {
        fprintf(stderr, "File \"%s\" does not have a max of 255.\n", filename1);
        exit(EXIT_FAILURE);
    }
 
    int  f2x, f2y, max2;
    fscanf(f2, "%s\n%d %d\n%d\n", p6, &f2x, &f2y, &max2);
    if (strcmp(p6, "P6") != 0)
    {
        fprintf(stderr, "File \"%s\" is not of type P6.\n", filename2);
        exit(EXIT_FAILURE);
    }
    if (max2 != 255)
    {
        fprintf(stderr, "File \"%s\" does not have a max of 255.\n", filename2);
        exit(EXIT_FAILURE);
    }
 
    if (f1x != f2x || f1y != f2y)
    {
        fprintf(stderr, "File \"%s\" has dimensions %d x %d, but file \"%s\" has dimensions %d x %d.  They cannot be equal.\n",
                        filename1, f1x, f1y, filename2, f2x, f2y);
        exit(EXIT_FAILURE);
    }

    int nx = f1x;  // could also be f2x
    int ny = f1y;  // could also be f2y
    int numpixels = nx*ny;
    int numbytes  = numpixels*3;
    unsigned char *buffer1 = (unsigned char *) malloc(numbytes);
    unsigned char *buffer2 = (unsigned char *) malloc(numbytes);
    fread(buffer1, sizeof(unsigned char), numbytes, f1);
    fread(buffer2, sizeof(unsigned char), numbytes, f2);

    int totalDiff = 0;
    for (int R = ny-1 ; R >= 0 ; R--)
    {
        for (int C = 0 ; C < nx ; C++)
        {
            int index = R*nx+C;
            int numdiff = 0;
            numdiff += (buffer1[3*index+0] == buffer2[3*index+0] ? 0 : 1);
            numdiff += (buffer1[3*index+1] == buffer2[3*index+1] ? 0 : 1);
            numdiff += (buffer1[3*index+2] == buffer2[3*index+2] ? 0 : 1);
            if (numdiff > 0)
            {
                printf("Difference at column %d, row %d (%d with top/bottom transformation): %s has (%d, %d, %d) while %s has (%d, %d, %d)\n", 
                        C, ny-R-1, R, 
                        filename1, buffer1[3*index+0], buffer1[3*index+1], buffer1[3*index+2],
                        filename2, buffer2[3*index+0], buffer2[3*index+1], buffer2[3*index+2]);
                totalDiff++;
/*
                if (totalDiff > 20)
                {
                    printf("Stopping print statements after 20 different pixels found.  There may be more pixels that are different.\n");
                    exit(EXIT_FAILURE);
                }
 */
            }
        }
    }
    if (totalDiff > 0)
    {
        printf("The number of different pixels is %d\n", totalDiff);
        exit(EXIT_FAILURE);
    }

    printf("The images are the same.\n");
    exit(EXIT_SUCCESS);
}
