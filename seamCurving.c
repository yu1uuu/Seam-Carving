#if !defined(SEAMCARVING_H)
#define SEAMCARVING_H

// #include <stdio.h>
// #include <stdlib.h>
#include <math.h>
// #include "seamcarving.h"
// #include "c_img.c"
#include "c_img.h"

void calc_rgb_val_x(struct rgb_img *im, int x, int x_a, int x_b, int y, int *R_x, int *G_x, int *B_x) {
    *R_x = get_pixel(im, y, x_a, 0) - get_pixel(im, y, x_b, 0);
    *G_x = get_pixel(im, y, x_a, 1) - get_pixel(im, y, x_b, 1);
    *B_x = get_pixel(im, y, x_a, 2) - get_pixel(im, y, x_b, 2);
}

void calc_rgb_val_y(struct rgb_img *im, int x, int y, int y_a, int y_b, int *R_y, int *G_y, int *B_y) {
    *R_y = get_pixel(im, y_a, x, 0) - get_pixel(im, y_b, x, 0);
    *G_y = get_pixel(im, y_a, x, 1) - get_pixel(im, y_b, x, 1); 
    *B_y = get_pixel(im, y_a, x, 2) - get_pixel(im, y_b, x, 2); 
}

void calc_energy(struct rgb_img *im, struct rgb_img **grad) {
    uint8_t dual_grd_energy;
    int R_x, G_x, B_x, R_y, G_y, B_y;
    double delta_x, delta_y;
    int x_bound = im->width - 1, y_bound = im->height - 1;
    create_img(grad, im->height, im->width);
    
    for (int i = 0; i < im->width; i++) {
        for (int j = 0; j < im->height; j++) {
            if (i == 0) {
                calc_rgb_val_x(im, i, 1, x_bound, j, &R_x, &G_x, &B_x);
            } else if (i == x_bound) {
                calc_rgb_val_x(im, i, 0, x_bound - 1, j, &R_x, &G_x, &B_x);
            } else {
                calc_rgb_val_x(im, i, i + 1, i - 1, j, &R_x, &G_x, &B_x);
            } if (j == 0) {
                calc_rgb_val_y(im, i, j, 1, y_bound, &R_y, &G_y, &B_y);
            } else if (j == y_bound) {
                calc_rgb_val_y(im, i, j, 0, y_bound - 1, &R_y, &G_y, &B_y);
            } else {
                calc_rgb_val_y(im, i, j, j + 1, j - 1, &R_y, &G_y, &B_y);
            }
            delta_x = (double)(R_x*R_x + G_x*G_x + B_x*B_x);
            delta_y = (double)(R_y*R_y + G_y*G_y + B_y*B_y);
            dual_grd_energy = (uint8_t)(sqrt(delta_x + delta_y) / 10); 
            set_pixel(*grad, j, i, dual_grd_energy, dual_grd_energy, dual_grd_energy);
        }
    }
}

void dynamic_seam(struct rgb_img *grad, double **best_arr) {
    *best_arr = (double *)malloc(sizeof(double) * grad->height * grad->width);
    int width = grad->width, height = grad->height;
    double cost, cost_left, cost_right;
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            if (j == 0) {
                (*best_arr)[i] = get_pixel(grad, j, i, 0);
            } else if (j > 0) {
                cost = get_pixel(grad, j, i, 0) + (*best_arr)[(j - 1)*width + i]; 
                if (i == 0) {
                    cost_right = get_pixel(grad, j, i, 0) + (*best_arr)[(j - 1) * width + i + 1]; 
                    (*best_arr)[j * width + i] = fmin(cost, cost_right); 
                } else if (i == width - 1) {
                    cost_left = get_pixel(grad, j, i, 0) + (*best_arr)[(j - 1) * width + i - 1];
                    (*best_arr)[j * width + i] = fmin(cost, cost_left);
                } else {
                    cost_left = get_pixel(grad, j, i, 0) + (*best_arr)[(j - 1) * width + i - 1];
                    cost_right = get_pixel(grad, j, i, 0) + (*best_arr)[(j - 1) * width + i + 1]; 
                    (*best_arr)[j * width + i] = fmin(fmin(cost_left, cost_right), cost);
                }
            }
        }
    }
}

void recover_path(double *best, int height, int width, int **path) {
    *path = (int *)malloc(sizeof(int) * height);
    int reverse_path[height]; 

    double min = best[(height - 1) * width];
    for (int i = 1; i < width; i++) {
        if (best[(height - 1) * width + i] < min) {
            min = best[(height - 1) * width + i]; 
            reverse_path[0] = i;
        }
    }
    
    double cost, cost_left, cost_right;
    int index; 
    for (int j = 1; j < height; j++) {
        index = reverse_path[j - 1];
        cost = best[(height - 1 - j) * width + index];
        if (index == 0) {
            cost_right = best[(height - j - 1) * width + index + 1];
            if (cost < cost_right) {
                reverse_path[j] = index;
            } else {
                reverse_path[j] = index + 1;
            }
        } else if (index == width - 1) {
            cost_left = best[(height - j - 1) * width + index - 1]; 
            if (cost < cost_left) {
                reverse_path[j] = index;
            } else {
                reverse_path[j] = index - 1;
            }
        } else {
            cost_left = best[(height - j - 1) * width + index - 1]; 
            cost_right = best[(height - j - 1) * width + index + 1];
            if (cost < cost_left && cost < cost_right) {
                reverse_path[j] = index;
            } else if (cost_left < cost && cost_left < cost_right) {
                reverse_path[j] = index - 1;
            } else if (cost_right < cost && cost_right < cost_left) {
                reverse_path[j] = index + 1;
            } else {
                reverse_path[j] = index;
            }
        } 
    }
    
    for (int i = 0; i < height; i++) {
        (*path)[i] = reverse_path[height - i - 1];
    } 
}

void remove_seam(struct rgb_img *src, struct rgb_img **dest, int *path) {
    int height = src->height, width = src->width - 1;
    int r, g, b;
    create_img(dest, height, width);
    for (int j = 0; j < height; j++) {
        int i = 0;
        while (i < path[j]) { 
            r = get_pixel(src, j, i, 0);
            g = get_pixel(src, j, i, 1);
            b = get_pixel(src, j, i, 2);
            set_pixel(*dest, j, i, r, g, b);
            i++;
        }
        i++;
        while (i < width) {
            r = get_pixel(src, j, i, 0);
            g = get_pixel(src, j, i, 1);
            b = get_pixel(src, j, i, 2);
            set_pixel(*dest, j, i-1, r, g, b);
            i++;            
        }
    }
}

/*
int main() {
    struct rgb_img *im;
    struct rgb_img *cur_im;
    struct rgb_img *grad;
    double *best;
    int *path;

    read_in_img(&im, "HJoceanSmall.bin");
    
    
    for(int i = 0; i < 150; i++){
        printf("i = %d\n", i);
        calc_energy(im,  &grad);
        dynamic_seam(grad, &best);
        recover_path(best, grad->height, grad->width, &path);
        remove_seam(im, &cur_im, path);

        char filename[200];
        sprintf(filename, "img%d.bin", i);
        write_img(cur_im, filename);


        destroy_image(im);
        destroy_image(grad);
        free(best);
        free(path);
        im = cur_im;
    }
    destroy_image(im);
    

}
*/

#endif
