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

    FILE *f1 = fopen(filename1, "rb");
    if (f1 == NULL)
    {
        fprintf(stderr, "Unable to open \"%s\" for reading\n", filename1);
        exit(EXIT_FAILURE);
    }
    fseek(f1, 0, SEEK_END);
    int filesize1 = ftell(f1);
    fseek(f1, 0, SEEK_SET);
    FILE *f2 = fopen(filename2, "rb");
    if (f2 == NULL)
    {
        fprintf(stderr, "Unable to open \"%s\" for reading\n", filename2);
        exit(EXIT_FAILURE);
    }
    fseek(f2, 0, SEEK_END);
    int filesize2 = ftell(f2);
    fseek(f2, 0, SEEK_SET);

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

    int curpos1 = ftell(f1);
    if (filesize1 != curpos1 + numbytes)
    {
        fprintf(stderr, "Possible corruption of \"%s\".  The header information took %d bytes and the data should be %d bytes for a total of %d bytes.  But the file \"%s\" has different number of bytes: %d.\n", filename1, curpos1, numbytes, curpos1+numbytes, filename1, filesize1);
        exit(EXIT_FAILURE);
    }

    int curpos2 = ftell(f2);
    if (filesize2 != curpos2 + numbytes)
    {
        fprintf(stderr, "Possible corruption of \"%s\".  The header information took %d bytes and the data should be %d bytes for a total of %d bytes.  But the file \"%s\" has a different number of bytes: %d.\n", filename2, curpos2, numbytes, curpos2+numbytes, filename2, filesize2);
        exit(EXIT_FAILURE);
    }

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
                totalDiff++;
            }
            else
            {
                int index_rev = (R)*nx+C;
            }
        }
    }
    if (totalDiff > 200)
    {
        printf("The number of different pixels is %d\n", totalDiff);
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
