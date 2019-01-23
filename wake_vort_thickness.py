contour = plt.contourf(xmesh, zmesh, ui[label1], vals, norm=norm, extend='max')#, vals, norm, cmap=cmap)
for c in contour.collections:
    c.set_edgecolor("face")
#plt.clim(minval, maxval)
triang = tri.Triangulation(xslice,zslice)
ax.tricontourf(triang, xslice, colors='black', zorder=10)
fig.savefig('umeans_contour.pdf')
fig.savefig('umeans_contour.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, max_u)
fig.savefig('umeans_max.pdf')
fig, ax = plt.subplots(1,1)
plt.plot(xi, min_u)
fig.savefig('umeans_min.pdf')
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, zmesh[max_grad_pos,:])
fig.savefig('max_grad_pos.pdf')
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, max_grad_u)
plt.ylim(0, 10000)
fig.savefig('max_grad.pdf')
plt.close()

fig, ax = plt.subplots(1, 1)
plt.plot(xi, vort_thick)
plt.ylim(0, 0.25)
plt.xlim(0, 0.115)
plt.ylabel(r'$\delta_{\omega}$', labelpad=-2)
plt.xlabel('x [m]')
plt.grid(True)
fig.savefig('vortthick.pdf')


xpos_list = [1.3, 1.4, 1.5, 1.6]
xref_list = [1, 1.5, 2, 2.5, 3]
#xref_list = [2]
xpos_list = (np.asarray(xref_list) * 0.189144) + 1
print xpos_list



#xi, zi, ui, uu, wi, ww, samp_u, samp_w = get_vert_line(xpos_list, x_WT, z_WT, u_WT, uu_WT, w_WT, ww_WT, samples_u, samples_w)
u_inf = 54.65

xpos_list = [50, 100, 150, 200, 249]
xpos_list = [20, 40, 60, 80, 99]

fig, ax = plt.subplots(1, len(xpos_list), figsize=(width,0.6*width), sharey=True)
for xpos in range(len(xpos_list)):
    #col = line_color.next()
    i = xpos_list[xpos]
    print i
    #print(ui[:,i].shape)
    #print(zi.shape)
    ax[xpos].plot(ui[label1][ind[i]:-1,i], zi[ind[i]:-1], label=str(np.max(ui[label1][:,i])))
    #ax[xpos].axhline(zi[ind[i]], color='k')
    ax[xpos].axhline(zi[max_grad_pos[i]], color='r')

    print(str(np.max(ui[label1][:,i])))
    print('pos of grad maximum: ' + str(max_grad_pos[i]))
#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
#    ax[i].set_xlim(0.3, 1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('umeans_'+plane+'.pdf')
plt.savefig('umeans_'+plane+'.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, len(xpos_list), figsize=(width,0.6*width), sharey=True)
for xpos in range(len(xpos_list)):
    #col = line_color.next()
    i = xpos_list[xpos]
    #print(ui[:,i].shape)
    #print(zi.shape)
    ax[xpos].plot(uy[ind[i]:-1,i], zi[ind[i]:-1], label=str(np.max(ui[label1][:,i])))
#    ax[xpos].axhline(zi[ind[i]])
#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[xpos].set_xlim(-1000, 15000)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('umeans_'+plane+'_gradients.pdf')
plt.savefig('umeans_'+plane+'_gradients.png', dpi=600)
plt.close()


'''
fig, ax = plt.subplots(1, 1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case, _ in files.iteritems():
    pos = np.argmin(np.where(ui[case]==0, ui[case].max(), ui[case]), axis=0)
    print('shape of xi:' + str(xi.shape))
    print('shape of zi:' + str(zi[pos].shape))
    plt.plot(xi, zi[pos], color=line_color.next(), label=case) # , linestyle=line_style.next(), marker=markers.next(), mew=1, ms=2, markevery=10)
    np.savez(case+'_' + plane + '_wake_min_pos', xi=xi, zi=zi[pos])

plt.legend(loc='best')

image_name = case_name+'_vt'
#    print("exporting image " + image_name)
plt.ylim(-0.05, 0.25)
fig.savefig(image_name + '.png', dpi=600)
plt.close(fig)



[max_u, index_max_u] = max(interp_u);
[min_u, index_min_u] = min(interp_u);

index_max_grad_u = linspace(0,0,n);
real_index_max_grad_u = linspace(0,0,n);
vort_thick = linspace(0,0,n);

delta_u = zeros(m,n);
delta_z = zeros(m,n);

for j=1:n

    for i=1:m-1
        delta_u(i,j) = (interp_u(i+1,j)-interp_u(i,j));
        delta_z(i,j) = (interp_z(i+1,j)-interp_z(i,j));
    end

end

for j=1:n
    [sel index_low_profile_bound] = max(interp_u==0, [], 1);
    [max_grad_u(j) index_max_grad_u(j)] = max(gradient(interp_u(index_low_profile_bound(j):end,j))/z_spacing);
    real_index_max_grad_u(j) = index_max_grad_u(j) + index_low_profile_bound(j);
    vort_thick(j) = ((max_u(j)-min_u(j))/max_grad_u(j))./data.c;

end

%gradient_u = delta_u./delta_z;
max_grad_zero_u = max(delta_u./delta_z);
vort_thick_zero_u = (max_u./max_grad_zero_u)./data.c;
'''
