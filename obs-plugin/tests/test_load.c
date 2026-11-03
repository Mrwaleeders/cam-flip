/*
 * Loads the built cam-flip.plugin into a real libobs exactly the way OBS does at startup,
 * and checks the filter is registered. Catches "compiled with newer libobs" refusals.
 *   test_load <path to cam-flip.plugin/Contents/MacOS/cam-flip>
 */
#include <obs.h>
#include <obs-module.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("usage: test_load <plugin binary>\n");
		return 2;
	}
	if (!obs_startup("en-US", NULL, NULL)) {
		printf("FAIL: obs_startup\n");
		return 1;
	}
	printf("libobs %s\n", obs_get_version_string());
	obs_module_t *module = NULL;
	int r = obs_open_module(&module, argv[1], "");
	if (r != MODULE_SUCCESS) {
		printf("FAIL: OBS refused to open the plugin (code %d)\n", r);
		return 1;
	}
	if (!obs_init_module(module)) {
		printf("FAIL: OBS refused to load the plugin (see log above)\n");
		return 1;
	}
	const char *name = obs_source_get_display_name("camflip_filter");
	if (!name) {
		printf("FAIL: camflip_filter not registered\n");
		return 1;
	}
	printf("ok: OBS loaded the plugin and registered \"%s\"\n", name);
	obs_shutdown();
	return 0;
}
