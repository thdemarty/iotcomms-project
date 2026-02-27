# Name of the application
APPLICATION = link-quality-fingerprint

# If no BOARD is found in the environment, use this default:
BOARD ?= adafruit-feather-nrf52840-sense
		 
# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= $(CURDIR)/RIOT

USEPKG += nimble
USEMODULE += nimble
USEMODULE += nimble_netif
USEMODULE += nimble_addr
USEMODULE += gnrc_netif
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc_neterr
USEMODULE += gnrc_pktdump
USEMODULE += gnrc_ipv6
USEMODULE += gnrc_icmpv6_echo
USEMODULE += ztimer
USEMODULE += ztimer_msec
USEMODULE += bluetil_ad
USEMODULE += ws281x

# Comment this out to disable code in RIOT that does safety checking
# which is not needed in a production environment but helps in the 
# development process:
DEVELHELP ?= 1

# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

NODEID ?= 0
NODE_COUNT ?= 3

# Specify NODEID when invoking make
CFLAGS += -DNODEID=$(NODEID)
CFLAGS += -DNODE_COUNT=$(NODE_COUNT)

# Set the maximum number of connections allowed by nimble
NIMBLE_MAX_CONN = 5

# Increase the default pktbuf size to be able to hold large throughput bursts of packets
CFLAGS += -DCONFIG_GNRC_PKTBUF_SIZE=65536
CFLAGS += -DMYNEWT_VAL_BLE_LL_TX_PWR_DBM=8
include $(RIOTBASE)/Makefile.include



