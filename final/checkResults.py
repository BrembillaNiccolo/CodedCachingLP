import re
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
import sys
from collections import Counter
from scipy.spatial import ConvexHull
from scipy.special import comb
from math import floor

folder = sys.argv[1]
filename = sys.argv[2] # Update with your actual file path
num_users = int(sys.argv[3]) if len(sys.argv) > 3 else 4
num_files = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(folder, exist_ok=True)


def convex_envelope(x, y, r, k):
    points = np.concatenate([np.array(list(zip(*[x, y]))), np.array([[r, min(r, k)]])])
    hull = ConvexHull(points)
    points = points[hull.vertices]
    points = points[np.lexsort([points[:,0], points[:,1]])][::-1]
    points = points[1:]     
    return points[:, 0], points[:, 1]

def thm2converse(k, n, convex=False):
    x, y = [0], [min(k,n)]
    for s in range(1, min(k,n)+1):
        for l in range(1, s+1):
            x += [(n-l+1)/s]
            y += [(s-1)/2+l*(l-1)/2/s]
    if not convex:
        return x, y
    return convex_envelope(x, y, n, k)

def thm4converse(k, n):
    As, Bs = [], []
    for p in range(max(1, k-n+1), k):
        alpha = floor((n-1)/(k-p))
        beta = n - alpha*(k-p)
        if beta + alpha*(k-2*p-1)/2 <= 0:
            As += [(2*k-p+1)/(p+1)]
            Bs += [k*(k+1)/p/(p+1)/n]
        else:
            As += [(2*k-p+1)/(p+1)]
            Bs += [2*k*(k-p)/p/(p+1)/(n-beta)]
            
    xs = np.linspace(0, n, 20*n+1)
    ys = []
    for x in xs:
        r = 0
        for i in range(len(As)):
            r = max(As[i] - Bs[i]*x, r)
        ys += [r]
    # print(As, Bs)
    return xs, ys


def achievability(n, k):
    ts = list(reversed(range(2, n+1)))
    xs = [0] + [n/k/t for t in ts]
    def f(t):
        tmp = k*comb(n-1, t-1)*(t-1) + (k-n)*comb(n-1,t-1)+comb(n,t+1)*t*t+comb(n-1,t)*t*(k-n)
        return tmp/k/comb(n-1,t-1)
    ys = [n] + [f(t) for t in ts]
    ts = [t for t in range(1,k)]
    xs += [n*comb(k-1,t-1)/comb(k,t) for t in ts]+[n]
    ys += [(comb(k,t+1)-comb(k-min(k,n),t+1))/comb(k,t) for t in ts]+[0]
    return convex_envelope(xs, ys, n, n)

def converse(n, k):
    xs = np.linspace(0, n, 20*n+1)
    _, ys_thm4converse = thm4converse(k, n)
    thm2converse_x, thm2converse_y = thm2converse(k, n, True)
    ys_thm2converse = np.interp(xs, thm2converse_x, thm2converse_y)
    ys = np.maximum(ys_thm2converse, ys_thm4converse)
    return xs, ys

def compute_y_on_segment(xA, yA, xB, yB, xC):
    if xA == xB:
        raise ValueError("The segment is vertical, cannot determine y from x.")
    yC = yA + (xC - xA) * (yB - yA) / (xB - xA)
    return yC

def parse_tradeoffs_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = {}
    original_names = {}  # Dictionary to store the original format
    all_M_values = set()
    i = 0
    
    while i < len(lines):
        match = re.match(r'Tradeoffs, N = \d+, K = \d+', lines[i].strip())
        if match:
            original_name = lines[i].strip()  # Store the original name
            formatted_name = f"Tradeoff {len(data)+1}"
            
            M = list(map(float, lines[i+1].strip('[]\n').split(',')))
            R = list(map(float, lines[i+2].strip('[]\n').split(',')))
            
            data[formatted_name] = (M, R)
            original_names[formatted_name] = original_name  # Store the original format
            all_M_values.update(M)
            
            i += 3
        else:
            i += 1

    all_M_values = sorted(all_M_values)

    for key, (M, R) in data.items():
        for i in range(len(all_M_values)):
            if all_M_values[i] not in M:
                # Find last element < all_M_values[i] and the next element > all_M_values[i]
                for j in range(len(M)):
                    if M[j] < all_M_values[i]:
                        left = j
                    else:
                        break
                right = left + 1
                r_value = compute_y_on_segment(M[left], R[left], M[right], R[right], all_M_values[i])
                #put between left and right
                M.insert(left + 1, all_M_values[i])
                R.insert(left + 1, r_value)
                
    return data, all_M_values, original_names

def find_highest_R(data, all_M_values, output_filename=folder+'/highest_R_values.txt'):
    os.makedirs(folder, exist_ok=True)
    highest_R_per_M = {M: [] for M in all_M_values}
    
    for key, (M, R) in data.items():
        for i, m_value in enumerate(M):
            if not highest_R_per_M[m_value] or R[i] > highest_R_per_M[m_value][0][1]:
                highest_R_per_M[m_value] = [(key, R[i])]
            elif R[i] == highest_R_per_M[m_value][0][1]:
                highest_R_per_M[m_value].append((key, R[i]))
    
    with open(output_filename, 'w') as f:
        for M, entries in highest_R_per_M.items():
            f.write(f"M = {M}: {', '.join([entry[0] for entry in entries])}\n")
    return highest_R_per_M


def getPositionPercentage(M, R, x_ach, y_ach, x_con, y_con):
    position_percentage = []
    M_Used = []
    lastPositionPercentage = 0
    isFirst = True
    for i in range(len(M)):
        #find the closest point on the line between achievability and converse bounds
        left = 0
        while x_ach[left+1] < M[i]:
            left += 1
        #if left is the lat point y_ach is 0
        if left == len(x_ach):
            left -= 1
            y_ach_real = 0
        else:
            right = left + 1
            y_ach_real = y_ach[left] + (y_ach[right] - y_ach[left])*(M[i] - x_ach[left])/(x_ach[right] - x_ach[left])
        while x_con[left+1] < M[i]:
            left += 1
        if left == len(x_con):
            left -= 1
            y_con_real = y_con[left]
        else:
            right = left + 1
            y_con_real = y_con[left] + (y_con[right] - y_con[left])*(M[i] - x_con[left])/(x_con[right] - x_con[left])
        #find the position percentage
        distance = round(y_ach_real - R[i], 6)
        max_distance = round(y_ach_real - y_con_real, 6)
        if( distance > max_distance and M[i] < 1):
            lastPositionPercentage = distance
            isFirst = False
            continue
        elif not isFirst:
            isFirst = True
            position_percentage.append(lastPositionPercentage)
            M_Used.append(M[i-1])

        position_percentage.append(distance)
        M_Used.append(M[i])
    return position_percentage,M_Used

def filter_and_save(data, all_M_values, original_names, output_filename=folder+'/unique_tradeoffs.txt', output_dir=folder+'/tradeoff_plots'):
    os.makedirs(output_dir, exist_ok=True)
    unique_values = {}
    duplicates = {}
    
    with open(folder+'/duplicate_tradeoffs.txt', 'w') as dup_file:
        for key, (M, R) in data.items():
            found = False
            for existing_key, values in unique_values.items():
                if values == (M, R):
                    duplicates.setdefault(existing_key, []).append(key)
                    found = True
                    break
            if not found:
                unique_values[key] = (M, R)
        
        for main_key, duplicate_keys in duplicates.items():
            dup_file.write(f"{main_key} is equal to {', '.join(duplicate_keys)}\n")
    
    with open(output_filename, 'w') as f:
        for i, (key, (M, R)) in enumerate(unique_values.items()):
            # Print the first line in its original format
            f.write(f"{original_names[key]}\n")
            f.write(f"{M}\n{R}\n")
    
    dictionaryHighestR = find_highest_R(unique_values, all_M_values)
    
    # Select most recurrent tradeoffs with the highest R values
    print("Starting the selection of the most recurrent elements.")
    
    # Find the frequency of each tradeoff (M, R)
    R_frequencies = {}
    mostImportantTradeoffs = []
    # Find the value in dictionaryHighestR that is the most present
    for M, entries in dictionaryHighestR.items():
        for entry in entries:
            key, R = entry
            R_frequencies[key] = R_frequencies.get(key, 0) + 1
        #if only one value is present, add it to mostImportantTradeoffs if not already present
        if len(entries) == 1 and entries[0][0] not in mostImportantTradeoffs:
            mostImportantTradeoffs.append(entries[0][0])  
    ordered_R_frequencies = sorted(R_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    

    for key, _ in ordered_R_frequencies:
        #check if for some M the R value is higher than the one of each element in mostImportantTradeoffs
        #if it is, add it to the list
        M, R = unique_values[key]
        toAdd = []
        for i in range(len(M)):
            toAdd.append(True)
        for tradeoff in mostImportantTradeoffs:
            M2, R2 = unique_values[tradeoff]
            for i in range(len(M)):
                for j in range(len(M2)):
                    if M[i] == M2[j] and R[i] <= R2[j]:
                        toAdd[i] = False
        #if at least one of the R values is higher, add the tradeoff to the list
        if any(toAdd):
            mostImportantTradeoffs.append(key)
    #recheck if there are any tradeoffs that are equal to the ones in mostImportantTradeoffs
    #if there are, remove them
    remainingTradeoffs = mostImportantTradeoffs.copy()
    print("Most important tradeoffs before filtering: ", mostImportantTradeoffs)
    with open(folder+'/mostImportantTradeoffsPositions.txt', 'w') as f:
        #do nothing, just create the file
        pass
    for tradeoff in mostImportantTradeoffs:
        M, R = unique_values[tradeoff]
        toRemove = []
        for i in range(len(M)):
            toRemove.append(False)
        for tradeoff2 in remainingTradeoffs:
            if tradeoff == tradeoff2:
                continue
            M2, R2 = unique_values[tradeoff2]
            for i in range(len(M)):
                for j in range(len(M2)):
                    if M[i] <= M2[j] and R[i] <= R2[j]:
                        toRemove[i] = True
                        break
                    if M[i] >= floor(M[-1]/2):
                        print(f"M[i] = {M[i]}, M/2 = {floor(M[-1]/2)}")
                        toRemove[i] = True
                        break
        if all(toRemove) :
            
            remainingTradeoffs.remove(tradeoff)
            print(f"Tradeoff {tradeoff} removed.")
        else:
            print(f"Tradeoff {tradeoff} kept,sum of toRemove: {sum(toRemove)}")
            #print on a file the tradeoff
            with open(folder+'/mostImportantTradeoffsPositions.txt', 'a') as f:
                f.write(f"{tradeoff}\n")
                #Print M and R values that are false in toRemove
                for i in range(len(M)):
                    if toRemove[i] == False:
                        f.write(f"{M[i]}\t{R[i]}\n")
                f.write("\n")
    mostImportantTradeoffs = remainingTradeoffs
    print("Most important tradeoffs : ", mostImportantTradeoffs)
    print("In total ", len(mostImportantTradeoffs), " tradeoffs.")
    #save the most important tradeoffs
    with open(folder+'/mostImportantTradeoffs.txt', 'w') as f:
        for key in mostImportantTradeoffs:
            f.write(f"{original_names[key]}\n")
            f.write(f"{unique_values[key][0]}\n{unique_values[key][1]}\n")
    # Plot the tradeoffs
    for key in mostImportantTradeoffs:
        M, R = unique_values[key]
        plt.plot(M, R, label=key)

    #Add achievability and converse bounds
    x_ach, y_ach = achievability(num_users, num_files)
    plt.plot(x_ach, y_ach, label="Achievability bound", linestyle='dotted')
    x_con, y_con = converse(num_users, num_files)
    plt.plot(x_con, y_con, label="Converse bound", linestyle='dotted')

    plt.xlabel('M')
    plt.ylabel('R')
    #Reduce legend size top right
    plt.legend(loc='center left', fontsize='x-small', bbox_to_anchor=(0.7, 0.7))
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'tradeoff_plot.png'),dpi = 450)
    
    print("Most important tradeoffs : ", mostImportantTradeoffs)
    print("In total ", len(mostImportantTradeoffs), " tradeoffs.")
    #Last plot reset plot
    plt.clf()
    for element in mostImportantTradeoffs:
        M, R = unique_values[element]
        plt.plot(M, R, label=element)
    plt.plot(x_ach, y_ach, label="Achievability bound", linestyle='dotted')
    plt.plot(x_con, y_con, label="Converse bound", linestyle='dotted')
    #Reduce legend size top right
    plt.legend(loc='center left', fontsize='x-small', bbox_to_anchor=(0.7, 0.7))
    plt.xlabel('M')
    plt.ylabel('R')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'tradeoff_plot_filtered.png'),dpi = 450)
    #save the most important tradeoffs
    with open(folder+'/mostImportantTradeoffsFiltered.txt', 'w') as f:
        for key in mostImportantTradeoffs:
            f.write(f"{original_names[key]}\n")
            f.write(f"{unique_values[key][0]}\n{unique_values[key][1]}\n")
    print("Tradeoffs filtered and saved.")

    #need a plot as distance bewteen achievability and converse bounds I want to plot the tradeoffs as a line between them (achievability as 0 and converse as 1)
    plt.clf()
    #plot the bounds
    #achievability is 0
    x_ach, y_ach = achievability(num_users, num_files)
    x_con, y_con = converse(num_users, num_files)
    distance = []
    for i in range(len(y_con)):
        #find x_ach closest to x_con[i] from left and right
        left = 0
        
        while x_ach[left+1] < x_con[i]:
            left += 1
        
        if left == len(x_ach):
            left -= 1
            y_ach_real = 0
        else:
            right = left + 1
            #it is a segment find real y_ach vale for x_con[i] y_ach left is higher than y_ach right
            y_ach_real = y_ach[left] + (y_ach[right] - y_ach[left])*(x_con[i] - x_ach[left])/(x_ach[right] - x_ach[left])
        #use first 6 decimal points
        distance.append(round(y_ach_real - y_con[i], 6))
    plt.plot(x_con, distance, label="SoTA Achievability vs Converse distance", linestyle='dashed')
    #for each tradeoff find the closest point on the line between achievability and converse bounds
    for key in mostImportantTradeoffs:
        M, R = unique_values[key]
        position_percentage,M_used = getPositionPercentage(M, R, x_ach, y_ach, x_con, y_con)
        #If any distance absolute >1 don't print
        if any(abs(i) > 1 for i in position_percentage):
            continue
        #plot the tradeoff using M on the x axis and position_percentage on the y axis
        if len(M_used) != len(M):
            #deleter first M_used and position_percentage
            M_used = M_used[1:]
            position_percentage = position_percentage[1:]
        plt.plot(M_used, position_percentage, label=key)
    plt.xlabel('M')
    plt.ylabel('Distance between Achievability and Converse bounds\n (lower is better)')
    #Reduce legend size top right
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'tradeoff_plot_distance.png'),dpi = 450)






# Example usage
data, all_M_values,original_names = parse_tradeoffs_file(filename)
filter_and_save(data, all_M_values,original_names)
